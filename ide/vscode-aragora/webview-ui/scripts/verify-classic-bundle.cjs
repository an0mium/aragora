const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const distDir = path.join(__dirname, '..', 'dist');
const expectedFiles = ['index.html', 'main.css', 'main.js', 'main.js.map'];

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

if (/^\s*import\s/m.test(mainJs) || /^\s*export\s/m.test(mainJs)) {
  throw new Error('main.js contains ESM import/export syntax');
}

new vm.Script(mainJs, { filename: 'main.js' });
