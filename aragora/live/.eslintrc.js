module.exports = {
  extends: ['next/core-web-vitals'],
  rules: {
    // Warn on console.log statements in production code
    // console.warn and console.error are allowed
    'no-console': ['warn', { allow: ['warn', 'error'] }],
  },
}
