const { createKubernetesBackend } = require('./kubernetes');
const { createVertexBackend } = require('./vertex');
const { createLocalProcessBackend } = require('./localProcess');

function buildBackendRegistry(config = {}) {
  const registry = new Map();
  const instances = [
    createKubernetesBackend(config.kubernetes || {}),
    createVertexBackend(config.vertex || {}),
    createLocalProcessBackend(config.local || {}),
  ];

  instances.forEach((backend) => {
    registry.set(backend.name, backend);
  });

  return registry;
}

module.exports = { buildBackendRegistry };
