const { spawn } = require('child_process');

function envMapToList(env = {}, hfToken) {
  const combined = { ...env };
  if (hfToken) {
    combined.HF_TOKEN = hfToken;
  }
  return Object.entries(combined).map(([name, value]) => ({
    name,
    value: String(value),
  }));
}

function parseJobId(text = '') {
  const match = text.match(/customJobs\/([a-zA-Z0-9-]+)/);
  return match ? match[1] : null;
}

function createVertexBackend(config = {}) {
  const project =
    config.project ||
    process.env.VERTEX_PROJECT ||
    process.env.GCP_PROJECT ||
    process.env.PROJECT_ID;
  const region = config.region || process.env.VERTEX_REGION || 'us-central1';
  const imageUri =
    config.imageUri || process.env.VERTEX_IMAGE_URI || process.env.IMAGE_URI;
  const machineType =
    config.machineType ||
    process.env.VERTEX_MACHINE_TYPE ||
    'a2-highgpu-1g';
  const acceleratorType =
    config.acceleratorType ||
    process.env.VERTEX_ACCELERATOR_TYPE ||
    'NVIDIA_TESLA_A100';
  const acceleratorCount =
    Number(config.acceleratorCount) ||
    Number(process.env.VERTEX_ACCELERATOR_COUNT || 1);
  const replicaCount =
    Number(config.replicaCount) ||
    Number(process.env.VERTEX_REPLICA_COUNT || 1);
  const gcloudBin = config.gcloudBin || process.env.GCLOUD_BIN || 'gcloud';
  const serviceAccount =
    config.serviceAccount || process.env.VERTEX_SERVICE_ACCOUNT;
  const network = config.network || process.env.VERTEX_NETWORK;
  const autoSubmit =
    config.autoSubmit ?? (process.env.VERTEX_SUBMIT === '1' ? true : false);

  return {
    name: 'vertex',
    validate: ({ cliOptions }) => {
      if (!project) {
        return 'Set VERTEX_PROJECT (or PROJECT_ID) for Vertex submissions';
      }
      if (!imageUri) {
        return 'Set VERTEX_IMAGE_URI to the trainer container image';
      }
      const dataset = cliOptions.dataset_file || '';
      if (!dataset.startsWith('gs://')) {
        return 'dataset_file must be a gs:// URI for Vertex jobs';
      }
      return null;
    },
    submit: async ({ cliArgs, hfToken, env = {}, helpers, jobName }) => {
      const workerPool = {
        machineSpec: {
          machineType,
          acceleratorType,
          acceleratorCount,
        },
        replicaCount,
        containerSpec: {
          imageUri,
          args: cliArgs,
          env: envMapToList(env, hfToken),
        },
      };

      const jobSpec = { workerPoolSpecs: [workerPool] };
      if (serviceAccount) {
        jobSpec.serviceAccount = serviceAccount;
      }
      if (network) {
        jobSpec.network = network;
      }

      const spec = {
        displayName: jobName,
        jobSpec,
      };

      const specString = JSON.stringify(spec, null, 2);
      helpers.appendLog('stdout', `Vertex CustomJob spec:\n${specString}\n`);
      helpers.setStatus('submitting');

      if (!autoSubmit) {
        helpers.appendLog(
          'stdout',
          'Auto-submit disabled (set VERTEX_SUBMIT=1 to call gcloud).'
        );
        helpers.setStatus('submitted');
        return;
      }

      let gcloudOutput = '';
      await new Promise((resolve, reject) => {
        const child = spawn(
          gcloudBin,
          [
            'ai',
            'custom-jobs',
            'create',
            `--project=${project}`,
            `--region=${region}`,
            `--display-name=${jobName}`,
            '--config=-',
          ],
          {
            env: { ...process.env, ...env },
          }
        );
        child.stdin.write(specString);
        child.stdin.end();

        child.stdout.on('data', (data) => {
          gcloudOutput += data.toString();
          helpers.appendLog('stdout', data);
        });
        child.stderr.on('data', (data) => helpers.appendLog('stderr', data));
        child.on('error', reject);
        child.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`gcloud custom-jobs create exited with ${code}`));
          }
        });
      });

      const jobId = parseJobId(gcloudOutput);
      if (jobId) {
        helpers.setExternalId(jobId);
      }
      helpers.setStatus('submitted');
    },
  };
}

module.exports = { createVertexBackend };
