const express = require('express');
const { v4: uuidv4 } = require('uuid');

const {
  prepareTrainerArgs,
  REPO_ROOT,
} = require('./src/request');
const { buildBackendRegistry } = require('./src/backends');

const app = express();
app.use(express.json({ limit: '1mb' }));

const MAX_LOG_CHARS = Number(process.env.MAX_LOG_CHARS || 1_000_000);
const DEFAULT_BACKEND = (process.env.DEFAULT_TRAINING_BACKEND ||
  process.env.DEFAULT_BACKEND ||
  'kubernetes').toLowerCase();

const jobs = new Map();
const backends = buildBackendRegistry();

function appendLog(job, chunk, stream) {
  const text = chunk.toString();
  job.logs.push({ timestamp: new Date().toISOString(), stream, message: text });
  job.logTail += text;
  if (job.logTail.length > MAX_LOG_CHARS) {
    job.logTail = job.logTail.slice(-MAX_LOG_CHARS);
  }
}

function setStatus(job, status) {
  job.status = status;
  if (!job.startedAt && status !== 'queued') {
    job.startedAt = new Date().toISOString();
  }
  if (status === 'succeeded' || status === 'failed') {
    job.finishedAt = new Date().toISOString();
  }
}

function summarizeJob(job) {
  return {
    id: job.id,
    name: job.name,
    backend: job.backend,
    externalId: job.externalId,
    status: job.status,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    exitCode: job.exitCode,
    command: job.command,
    args: job.args,
    outputDir: job.outputDir,
    datasetFile: job.datasetFile,
    error: job.error,
    logTail: job.logTail,
  };
}

function availableBackends() {
  return Array.from(backends.keys());
}

function createHelpers(job) {
  return {
    appendLog: (stream, message) => appendLog(job, message, stream),
    setStatus: (status) => setStatus(job, status),
    finish: (status) => setStatus(job, status),
    setExitCode: (code) => {
      job.exitCode = code;
    },
    setCommand: (command) => {
      job.command = command;
    },
    setError: (message) => {
      job.error = message;
    },
    getError: () => job.error,
    setExternalId: (externalId) => {
      job.externalId = externalId;
    },
  };
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    repoRoot: REPO_ROOT,
    defaultBackend: DEFAULT_BACKEND,
    backends: availableBackends(),
  });
});

app.get('/jobs', (req, res) => {
  const list = Array.from(jobs.values()).map(summarizeJob);
  res.json({ jobs: list });
});

app.get('/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  res.json({ job: { ...summarizeJob(job), logs: job.logs } });
});

app.post('/train', (req, res) => {
  const payload = req.body || {};
  const requestedBackend =
    (payload.backend || DEFAULT_BACKEND || '').toLowerCase();
  const backend = backends.get(requestedBackend);
  if (!backend) {
    return res.status(400).json({
      error: `Unsupported backend "${requestedBackend}". Available: ${availableBackends().join(', ')}`,
    });
  }

  const id = uuidv4();
  const jobName = payload.job_name || payload.jobName || id;

  let trainerArgs;
  try {
    trainerArgs = prepareTrainerArgs(payload, { jobName });
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }

  const validationError = backend.validate
    ? backend.validate({ payload, ...trainerArgs })
    : null;
  if (validationError) {
    return res.status(400).json({ error: validationError });
  }

  const job = {
    id,
    name: jobName,
    backend: backend.name,
    externalId: null,
    status: 'queued',
    createdAt: new Date().toISOString(),
    startedAt: null,
    finishedAt: null,
    exitCode: null,
    command: backend.name,
    args: trainerArgs.cliOptions,
    datasetFile: trainerArgs.datasetFile,
    outputDir: trainerArgs.outputDir,
    logs: [],
    logTail: '',
    error: null,
  };
  jobs.set(id, job);

  res.status(202).json({ job: summarizeJob(job) });

  const helpers = createHelpers(job);
  const env = payload.env || {};

  Promise.resolve()
    .then(() =>
      backend.submit({
        jobName,
        cliArgs: trainerArgs.cliArgs,
        cliOptions: trainerArgs.cliOptions,
        hfToken: trainerArgs.hfToken,
        env,
        helpers,
        payload,
      })
    )
    .catch((err) => {
      helpers.appendLog('stderr', err.message);
      helpers.setError(err.message);
      helpers.finish('failed');
    });
});

const port = Number(process.env.PORT || 4000);
app.listen(port, () => {
  console.log(`Training API listening on port ${port}`);
  console.log(`Repo root: ${REPO_ROOT}`);
  console.log(`Default backend: ${DEFAULT_BACKEND}`);
});
