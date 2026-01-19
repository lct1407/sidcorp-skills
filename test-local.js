#!/usr/bin/env node
// Quick test script - run from any directory:
// node d:\Sources\sidcorp-skills\test-local.js lct1407/sidcorp-skills

process.chdir(__dirname);
process.chdir('cli');

// Install deps if missing
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

if (!fs.existsSync(path.join(__dirname, 'cli', 'node_modules'))) {
  console.log('Installing dependencies...');
  execSync('npm install', { stdio: 'inherit' });
}

// Change back to original cwd
process.chdir(process.env.INIT_CWD || process.cwd());

// Run the CLI
require('./cli/index.js');
