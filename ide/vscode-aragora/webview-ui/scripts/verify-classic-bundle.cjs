const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const distDir = path.join(__dirname, '..', 'dist');
const expectedFiles = ['main.css', 'main.js', 'main.js.map'];

for (const fileName of expectedFiles) {
  const filePath = path.join(distDir, fileName);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing webview build artifact: ${fileName}`);
  }
}

const unexpectedJs = fs
  .readdirSync(distDir)
  .filter((fileName) => fileName.endsWith('.js') && fileName !== 'main.js');

if (unexpectedJs.length > 0) {
  throw new Error(`Unexpected additional JavaScript chunk(s): ${unexpectedJs.join(', ')}`);
}

const mainJs = fs.readFileSync(path.join(distDir, 'main.js'), 'utf8');

if (/\bimport\.meta\b/.test(mainJs)) {
  throw new Error('main.js contains import.meta and is not classic-script safe');
}

if (/(^|[;\n\r])\s*import\s*(?:["'{*]|[A-Za-z_$])/m.test(mainJs)) {
  throw new Error('main.js contains static ESM import syntax');
}

if (/(^|[;\n\r])\s*export\b/m.test(mainJs)) {
  throw new Error('main.js contains ESM import/export syntax');
}

new vm.Script(mainJs, { filename: 'main.js' });

if (!/^(?:var\s+AragoraWebview\s*=|\(?function\s*\(|!\s*function\s*\()/.test(mainJs.trimStart())) {
  throw new Error('main.js is not emitted as an IIFE classic-script bundle');
}
