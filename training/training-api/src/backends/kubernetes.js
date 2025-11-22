const { spawn } = require('child_process');
const { URI_PREFIXES } = require('../request');

function parseJsonEnv(value) {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch (err) {
    return null;
  }
}

function toK8sName(name, fallback = 'training-job') {
  const clean = name
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, '-')
    .replace(/^-+/, '')
    .replace(/-+$/, '')
    .slice(0, 63);
  return clean || fallback;
}

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

function buildManifest({
  jobName,
  image,
  cliArgs,
  env,
  namespace,
  serviceAccountName,
  nodeSelector,
  tolerations,
  imagePullSecrets,
  backoffLimit,
  restartPolicy,
  ttlSecondsAfterFinished,
  resourceLimits,
}) {
  const container = {
    name: 'trainer',
    image,
    args: cliArgs,
    env,
  };

  if (resourceLimits && Object.keys(resourceLimits).length > 0) {
    container.resources = { limits: resourceLimits };
  }

  const podSpec = {
    restartPolicy: restartPolicy || 'Never',
    containers: [container],
  };

  if (serviceAccountName) {
    podSpec.serviceAccountName = serviceAccountName;
  }
  if (nodeSelector) {
    podSpec.nodeSelector = nodeSelector;
  }
  if (Array.isArray(tolerations) && tolerations.length > 0) {
    podSpec.tolerations = tolerations;
  }
  if (Array.isArray(imagePullSecrets) && imagePullSecrets.length > 0) {
    podSpec.imagePullSecrets = imagePullSecrets.map((name) =>
      typeof name === 'string' ? { name } : name
    );
  }

  const spec = {
    backoffLimit: backoffLimit ?? 0,
    template: { spec: podSpec },
  };

  if (ttlSecondsAfterFinished !== undefined) {
    spec.ttlSecondsAfterFinished = ttlSecondsAfterFinished;
  }

  return {
    apiVersion: 'batch/v1',
    kind: 'Job',
    metadata: {
      name: toK8sName(jobName),
      namespace,
    },
    spec,
  };
}

function createKubernetesBackend(config = {}) {
  const image = config.image || process.env.K8S_TRAINING_IMAGE;
  const namespace = config.namespace || process.env.K8S_NAMESPACE || 'default';
  const serviceAccountName =
    config.serviceAccountName || process.env.K8S_SERVICE_ACCOUNT;
  const kubectlBin = config.kubectlBin || process.env.KUBECTL_BIN || 'kubectl';
  const autoSubmit =
    config.autoSubmit !== undefined
      ? Boolean(config.autoSubmit)
      : process.env.K8S_APPLY === '1';
  const nodeSelector =
    config.nodeSelector ||
    parseJsonEnv(process.env.K8S_NODE_SELECTOR || process.env.K8S_NODE_SELECTOR_JSON);
  const tolerations =
    config.tolerations ||
    parseJsonEnv(process.env.K8S_TOLERATIONS || process.env.K8S_TOLERATIONS_JSON) ||
    [];
  const imagePullSecrets =
    config.imagePullSecrets ||
    parseJsonEnv(
      process.env.K8S_IMAGE_PULL_SECRETS ||
        process.env.K8S_IMAGE_PULL_SECRETS_JSON
    ) ||
    [];
  const backoffLimit = config.backoffLimit ?? Number(process.env.K8S_BACKOFF_LIMIT || 0);
  const ttlSecondsAfterFinished =
    config.ttlSecondsAfterFinished ??
    (process.env.K8S_TTL_SECONDS_AFTER_FINISHED
      ? Number(process.env.K8S_TTL_SECONDS_AFTER_FINISHED)
      : undefined);
  const restartPolicy =
    config.restartPolicy || process.env.K8S_RESTART_POLICY || 'Never';
  const resourceLimits =
    config.resourceLimits ||
    parseJsonEnv(process.env.K8S_RESOURCE_LIMITS || process.env.K8S_LIMITS_JSON);
  const kubeconfig =
    config.kubeconfig || process.env.K8S_KUBECONFIG || process.env.KUBECONFIG;
  const context = config.context || process.env.K8S_CONTEXT;

  return {
    name: 'kubernetes',
    validate: ({ cliOptions }) => {
      if (!image) {
        return 'Set K8S_TRAINING_IMAGE to the container image that runs train.py';
      }
      const hasRemoteInput =
        typeof cliOptions.dataset_file === 'string' &&
        URI_PREFIXES.some((prefix) =>
          cliOptions.dataset_file.toLowerCase().startsWith(prefix)
        );
      if (!hasRemoteInput) {
        return 'dataset_file must be a GCS or S3 URI for Kubernetes jobs';
      }
      return null;
    },
    submit: async ({ cliArgs, hfToken, env = {}, helpers, jobName }) => {
      const manifest = buildManifest({
        jobName,
        image,
        cliArgs,
        env: envMapToList(env, hfToken),
        namespace,
        serviceAccountName,
        nodeSelector,
        tolerations,
        imagePullSecrets,
        backoffLimit,
        restartPolicy,
        ttlSecondsAfterFinished,
        resourceLimits,
      });

      helpers.setStatus('submitting');
      helpers.setExternalId(manifest.metadata.name);

      const manifestString = JSON.stringify(manifest, null, 2);
      helpers.appendLog(
        'stdout',
        `Kubernetes Job manifest:\n${manifestString}\n`
      );

      if (!autoSubmit) {
        helpers.appendLog(
          'stdout',
          'Auto-apply disabled (set K8S_APPLY=1 to kubectl apply from the API).'
        );
        helpers.setStatus('submitted');
        return;
      }

      await new Promise((resolve, reject) => {
        const args = ['apply', '-f', '-'];
        if (namespace) {
          args.push('-n', namespace);
        }
        if (context) {
          args.push('--context', context);
        }
        const child = spawn(kubectlBin, args, {
          env: kubeconfig
            ? { ...process.env, KUBECONFIG: kubeconfig }
            : process.env,
        });
        child.stdin.write(manifestString);
        child.stdin.end();

        child.stdout.on('data', (data) => helpers.appendLog('stdout', data));
        child.stderr.on('data', (data) => helpers.appendLog('stderr', data));
        child.on('error', reject);
        child.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`kubectl apply exited with code ${code}`));
          }
        });
      });

      helpers.setStatus('submitted');
    },
  };
}

module.exports = { createKubernetesBackend };
