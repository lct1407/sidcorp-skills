#!/usr/bin/env node

const fs = require('fs-extra');
const path = require('path');
const https = require('https');
const http = require('http');
const tar = require('tar');
const prompts = require('prompts');
const os = require('os');

const AGENTS = [
  { id: 'claude', name: 'Claude', folder: '.claude' },
  { id: 'antigraphity', name: 'Antigraphity', folder: '.agent' }
];

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('Usage: add-skill <owner/repo> [--ref <branch|tag>]');
    console.log('Example: add-skill sidcorp/sidcorp-skills');
    console.log('         add-skill sidcorp/sidcorp-skills --ref develop');
    process.exit(1);
  }

  let repoArg = args[0];
  let ref = 'main';

  // Parse --ref flag
  const refIndex = args.indexOf('--ref');
  if (refIndex !== -1 && args[refIndex + 1]) {
    ref = args[refIndex + 1];
  }

  // Parse owner/repo from various formats
  const { owner, repo } = parseRepoArg(repoArg);

  if (!owner || !repo) {
    console.error('Invalid repository format. Use: owner/repo or GitHub URL');
    process.exit(1);
  }

  console.log(`\n📦 Fetching ${owner}/${repo} (ref: ${ref})...\n`);

  const tempDir = path.join(os.tmpdir(), `add-skill-${Date.now()}`);

  try {
    // Download and extract tarball
    await downloadAndExtract(owner, repo, ref, tempDir);

    // Find extracted folder (GitHub adds repo-ref suffix)
    const extractedFolders = await fs.readdir(tempDir);
    const repoFolder = extractedFolders[0];
    const repoPath = path.join(tempDir, repoFolder);

    // Read manifest
    const manifestPath = path.join(repoPath, 'manifest.json');
    if (!await fs.pathExists(manifestPath)) {
      console.error('manifest.json not found in repository root');
      process.exit(1);
    }

    const manifest = await fs.readJson(manifestPath);

    if (!manifest.skills || manifest.skills.length === 0) {
      console.error('No skills found in manifest.json');
      process.exit(1);
    }

    // Step 1: Select skills
    const skillChoices = manifest.skills.map(skill => ({
      title: `${skill.name} - ${skill.description}`,
      value: skill.id
    }));

    const skillsResponse = await prompts({
      type: 'multiselect',
      name: 'selectedSkills',
      message: 'Select skills to install',
      choices: skillChoices,
      min: 1,
      hint: '- Space to select, Enter to confirm'
    });

    if (!skillsResponse.selectedSkills || skillsResponse.selectedSkills.length === 0) {
      console.log('No skills selected. Exiting.');
      process.exit(0);
    }

    // Step 2: Detect and select agents
    const cwd = process.cwd();
    const detectedAgents = [];

    for (const agent of AGENTS) {
      const agentPath = path.join(cwd, agent.folder);
      if (await fs.pathExists(agentPath)) {
        detectedAgents.push(agent.id);
      }
    }

    const agentChoices = AGENTS.map(agent => ({
      title: `${agent.name} (${agent.folder})`,
      value: agent.id,
      selected: detectedAgents.includes(agent.id) || detectedAgents.length === 0
    }));

    const agentsResponse = await prompts({
      type: 'multiselect',
      name: 'selectedAgents',
      message: 'Select target agents',
      choices: agentChoices,
      min: 1,
      hint: detectedAgents.length > 0
        ? `- Detected: ${detectedAgents.join(', ')}`
        : '- No agents detected, select manually'
    });

    if (!agentsResponse.selectedAgents || agentsResponse.selectedAgents.length === 0) {
      console.log('No agents selected. Exiting.');
      process.exit(0);
    }

    // Step 3: Install skills
    const installed = [];

    for (const skillId of skillsResponse.selectedSkills) {
      const skill = manifest.skills.find(s => s.id === skillId);
      const skillSourcePath = path.join(repoPath, skill.path);

      if (!await fs.pathExists(skillSourcePath)) {
        console.warn(`⚠️  Skill path not found: ${skill.path}`);
        continue;
      }

      for (const agentId of agentsResponse.selectedAgents) {
        const agent = AGENTS.find(a => a.id === agentId);
        const destPath = path.join(cwd, agent.folder, 'skills', skillId);

        // Remove existing and copy fresh
        await fs.remove(destPath);
        await fs.ensureDir(path.dirname(destPath));
        await fs.copy(skillSourcePath, destPath);

        installed.push({
          skill: skill.name,
          agent: agent.name,
          path: destPath
        });
      }
    }

    // Step 4: Print summary
    console.log('\n✅ Installation Summary\n');
    console.log('─'.repeat(60));

    for (const item of installed) {
      const relativePath = path.relative(cwd, item.path);
      console.log(`  ${item.skill} → ${item.agent}`);
      console.log(`    📁 ${relativePath}`);
    }

    console.log('─'.repeat(60));
    console.log(`\n🎉 Installed ${installed.length} skill(s) successfully!\n`);

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  } finally {
    // Cleanup temp directory
    await fs.remove(tempDir).catch(() => {});
  }
}

function parseRepoArg(arg) {
  // Handle GitHub URLs
  if (arg.includes('github.com')) {
    const match = arg.match(/github\.com\/([^\/]+)\/([^\/\.\s]+)/);
    if (match) {
      return { owner: match[1], repo: match[2] };
    }
  }

  // Handle owner/repo format
  if (arg.includes('/')) {
    const [owner, repo] = arg.split('/');
    return { owner, repo: repo.replace(/\.git$/, '') };
  }

  return { owner: null, repo: null };
}

function downloadAndExtract(owner, repo, ref, destDir) {
  return new Promise((resolve, reject) => {
    const url = `https://codeload.github.com/${owner}/${repo}/tar.gz/${ref}`;

    const request = https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Handle redirect
        const redirectUrl = response.headers.location;
        const protocol = redirectUrl.startsWith('https') ? https : http;
        protocol.get(redirectUrl, handleResponse).on('error', reject);
        return;
      }

      handleResponse(response);
    });

    request.on('error', reject);

    function handleResponse(response) {
      if (response.statusCode === 404) {
        reject(new Error(`Repository not found: ${owner}/${repo} (ref: ${ref})`));
        return;
      }

      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download: HTTP ${response.statusCode}`));
        return;
      }

      fs.ensureDirSync(destDir);

      response
        .pipe(tar.x({ cwd: destDir }))
        .on('finish', resolve)
        .on('error', reject);
    }
  });
}

main();
