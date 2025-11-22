const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const URI_PREFIXES = ['gs://', 's3://'];
const RESERVED_KEYS = new Set([
  'job_name',
  'env',
  'trainer_args',
  'backend',
  'backend_args',
  'backend_params',
]);

function normalizePath(value) {
  if (typeof value !== 'string' || value.length === 0) {
    return value;
  }
  if (URI_PREFIXES.some((prefix) => value.startsWith(prefix))) {
    return value;
  }
  if (value.match(/^[a-zA-Z]+:\/\//)) {
    return value;
  }
  if (path.isAbsolute(value)) {
    return value;
  }
  return path.join(REPO_ROOT, value);
}

function toFlag(key) {
  return `--${key.replace(/_/g, '-')}`;
}

function collectTrainerOptions(payload = {}) {
  const cliOptions = { ...(payload.trainer_args || {}) };

  Object.entries(payload).forEach(([key, value]) => {
    if (RESERVED_KEYS.has(key)) {
      return;
    }
    cliOptions[key] = value;
  });

  // Support model aliases so callers can pass `model` or `model_name`.
  if (!cliOptions.model_id) {
    if (payload.model) {
      cliOptions.model_id = payload.model;
    } else if (payload.model_name) {
      cliOptions.model_id = payload.model_name;
    }
  }

  return cliOptions;
}

function buildCliArgs(cliOptions) {
  const args = [];
  Object.entries(cliOptions).forEach(([key, value]) => {
    if (value === undefined || value === null) {
      return;
    }
    if (typeof value === 'boolean') {
      if (value) {
        args.push(toFlag(key));
      }
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item) => {
        args.push(toFlag(key), String(item));
      });
      return;
    }
    args.push(toFlag(key), String(value));
  });
  return args;
}

function prepareTrainerArgs(payload, { jobName } = {}) {
  const cliOptions = collectTrainerOptions(payload);

  if (!cliOptions.dataset_file) {
    throw new Error('dataset_file is required');
  }
  if (!cliOptions.output_dir) {
    throw new Error('output_dir is required');
  }

  const hfToken = cliOptions.hf_token;
  if (hfToken) {
    delete cliOptions.hf_token;
  }

  cliOptions.dataset_file = normalizePath(cliOptions.dataset_file);
  cliOptions.output_dir = normalizePath(cliOptions.output_dir);

  // If callers pass an object for trainer_args we merge it, so cache the final view.
  const normalizedOptions = { ...cliOptions };
  const cliArgs = buildCliArgs(cliOptions);

  return {
    cliOptions: normalizedOptions,
    cliArgs,
    hfToken,
    datasetFile: normalizedOptions.dataset_file,
    outputDir: normalizedOptions.output_dir,
    jobName,
  };
}

module.exports = {
  REPO_ROOT,
  RESERVED_KEYS,
  URI_PREFIXES,
  normalizePath,
  toFlag,
  prepareTrainerArgs,
  collectTrainerOptions,
  buildCliArgs,
};
