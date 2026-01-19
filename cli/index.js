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

// Claude-specific install targets
const CLAUDE_TARGETS = [
  { id: 'skills', name: 'Skills', folder: 'skills' },
  { id: 'subagents', name: 'Subagents', folder: 'subagents' },
  { id: 'commands', name: 'Commands', folder: 'commands' }
];

const INSTALLED_MANIFEST = '.skill-manifest.json';

// Default skill repositories
const DEFAULT_REPOS = [
  { id: 'sidcorp/sidcorp-skills', name: 'Sidcorp Skills', description: 'Official skills collection' }
];

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  // Handle init command
  if (command === 'init') {
    await initSkills();
    return;
  }

  // Handle upgrade command
  if (command === 'upgrade') {
    await upgradeAll();
    return;
  }

  // Handle --update flag (legacy, same as upgrade --skills)
  if (args.includes('--update') || args.includes('-u')) {
    await updateInstalledSkills();
    return;
  }

  // Handle --list flag
  if (args.includes('--list') || args.includes('-l')) {
    await listInstalledSkills();
    return;
  }

  if (args.length === 0) {
    printUsage();
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

    // Step 1: Select items to install
    const installItems = await selectItemsToInstall(manifest);
    if (installItems.length === 0) {
      console.log('Nothing selected. Exiting.');
      process.exit(0);
    }

    // Step 2: Detect and select agents
    const cwd = process.cwd();
    const selectedAgents = await selectAgents(cwd);
    if (selectedAgents.length === 0) {
      console.log('No agents selected. Exiting.');
      process.exit(0);
    }

    // Step 3: For Claude agent, ask about targets (skills/subagents/commands)
    let claudeTargets = ['skills']; // default
    if (selectedAgents.includes('claude')) {
      claudeTargets = await selectClaudeTargets(installItems);
    }

    // Step 4: Install items
    const installed = await installItems_(
      installItems,
      selectedAgents,
      claudeTargets,
      repoPath,
      manifest,
      cwd
    );

    // Step 5: Save installation manifest for updates
    await saveInstalledManifest(cwd, {
      source: `${owner}/${repo}`,
      ref,
      version: manifest.version,
      installedAt: new Date().toISOString(),
      items: installed
    });

    // Step 6: Print summary
    printSummary(installed, cwd);

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  } finally {
    // Cleanup temp directory
    await fs.remove(tempDir).catch(() => {});
  }
}

async function initSkills() {
  console.log('\n🚀 Add-Skill Initialization\n');

  // Step 1: Ask for repository
  const repoResponse = await prompts({
    type: 'select',
    name: 'repo',
    message: 'Select skill repository to install from',
    choices: [
      ...DEFAULT_REPOS.map(r => ({
        title: `${r.name} (${r.id})`,
        description: r.description,
        value: r.id
      })),
      {
        title: 'Custom repository',
        description: 'Enter owner/repo manually',
        value: 'custom'
      }
    ]
  });

  if (!repoResponse.repo) {
    console.log('Cancelled.');
    process.exit(0);
  }

  let repoArg = repoResponse.repo;

  if (repoArg === 'custom') {
    const customResponse = await prompts({
      type: 'text',
      name: 'repo',
      message: 'Enter repository (owner/repo)',
      validate: value => value.includes('/') ? true : 'Format: owner/repo'
    });

    if (!customResponse.repo) {
      console.log('Cancelled.');
      process.exit(0);
    }

    repoArg = customResponse.repo;
  }

  // Now run normal install flow
  const { owner, repo } = parseRepoArg(repoArg);
  const ref = 'main';

  console.log(`\n📦 Fetching ${owner}/${repo}...\n`);

  const tempDir = path.join(os.tmpdir(), `add-skill-${Date.now()}`);

  try {
    await downloadAndExtract(owner, repo, ref, tempDir);

    const extractedFolders = await fs.readdir(tempDir);
    const repoFolder = extractedFolders[0];
    const repoPath = path.join(tempDir, repoFolder);

    const manifestPath = path.join(repoPath, 'manifest.json');
    if (!await fs.pathExists(manifestPath)) {
      console.error('manifest.json not found in repository root');
      process.exit(1);
    }

    const manifest = await fs.readJson(manifestPath);
    const cwd = process.cwd();

    const installItems = await selectItemsToInstall(manifest);
    if (installItems.length === 0) {
      console.log('Nothing selected. Exiting.');
      process.exit(0);
    }

    const selectedAgents = await selectAgents(cwd);
    if (selectedAgents.length === 0) {
      console.log('No agents selected. Exiting.');
      process.exit(0);
    }

    let claudeTargets = ['skills'];
    if (selectedAgents.includes('claude')) {
      claudeTargets = await selectClaudeTargets(installItems);
    }

    const installed = await installItems_(
      installItems,
      selectedAgents,
      claudeTargets,
      repoPath,
      manifest,
      cwd
    );

    await saveInstalledManifest(cwd, {
      source: `${owner}/${repo}`,
      ref,
      version: manifest.version,
      installedAt: new Date().toISOString(),
      items: installed
    });

    printSummary(installed, cwd);

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  } finally {
    await fs.remove(tempDir).catch(() => {});
  }
}

async function upgradeAll() {
  console.log('\n🔄 Upgrading Add-Skill...\n');

  // Step 1: Check for CLI update
  console.log('📦 Checking for CLI updates...');

  try {
    const { execSync } = require('child_process');

    // Get current version
    const packageJson = require('./package.json');
    const currentVersion = packageJson.version;

    // Check latest version on npm
    let latestVersion;
    try {
      const npmInfo = execSync('npm view add-skill version', { encoding: 'utf-8' }).trim();
      latestVersion = npmInfo;
    } catch (e) {
      console.log('   ⚠️  Could not check npm for updates\n');
      latestVersion = currentVersion;
    }

    if (latestVersion && latestVersion !== currentVersion) {
      console.log(`   Current: v${currentVersion}`);
      console.log(`   Latest:  v${latestVersion}`);

      const updateResponse = await prompts({
        type: 'confirm',
        name: 'update',
        message: 'Update CLI to latest version?',
        initial: true
      });

      if (updateResponse.update) {
        console.log('\n   Installing latest version...');
        try {
          execSync('npm install -g add-skill@latest', { stdio: 'inherit' });
          console.log('   ✅ CLI updated successfully!\n');
        } catch (e) {
          console.log('   ❌ Failed to update CLI. Try: npm install -g add-skill@latest\n');
        }
      }
    } else {
      console.log(`   ✓ CLI is up to date (v${currentVersion})\n`);
    }

  } catch (error) {
    console.log(`   ⚠️  Could not check CLI version: ${error.message}\n`);
  }

  // Step 2: Update installed skills
  console.log('📦 Checking for skill updates...\n');
  await updateInstalledSkills();
}

function printUsage() {
  console.log(`
Usage: sc <command> [options]

Commands:
  sc init                    Initialize and install skills (interactive)
  sc upgrade                 Update CLI and all installed skills
  sc <owner/repo>            Install skills from GitHub repository
  sc --list, -l              List installed skills

Options:
  --ref <branch|tag>         Use specific branch or tag (default: main)

Examples:
  sc init
  sc upgrade
  sc sidcorp/sidcorp-skills
  sc sidcorp/sidcorp-skills --ref develop
  sc --list
`);
}

async function selectItemsToInstall(manifest) {
  const allItems = [];

  // Add skills
  if (manifest.skills?.length > 0) {
    manifest.skills.forEach(skill => {
      allItems.push({
        type: 'skill',
        ...skill,
        title: `[Skill] ${skill.name} - ${skill.description}`
      });
    });
  }

  // Add subagents
  if (manifest.subagents?.length > 0) {
    manifest.subagents.forEach(subagent => {
      allItems.push({
        type: 'subagent',
        ...subagent,
        title: `[Subagent] ${subagent.name} - ${subagent.description}`
      });
    });
  }

  // Add commands
  if (manifest.commands?.length > 0) {
    manifest.commands.forEach(command => {
      allItems.push({
        type: 'command',
        ...command,
        title: `[Command] ${command.name} - ${command.description}`
      });
    });
  }

  if (allItems.length === 0) {
    console.error('No installable items found in manifest.json');
    process.exit(1);
  }

  const choices = allItems.map(item => ({
    title: item.title,
    value: item
  }));

  const response = await prompts({
    type: 'multiselect',
    name: 'selected',
    message: 'Select items to install',
    choices,
    min: 1,
    hint: '- Space to select, Enter to confirm'
  });

  return response.selected || [];
}

async function selectAgents(cwd) {
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

  const response = await prompts({
    type: 'multiselect',
    name: 'selectedAgents',
    message: 'Select target agents',
    choices: agentChoices,
    min: 1,
    hint: detectedAgents.length > 0
      ? `- Detected: ${detectedAgents.join(', ')}`
      : '- No agents detected, select manually'
  });

  return response.selectedAgents || [];
}

async function selectClaudeTargets(installItems) {
  // Check what types of items are being installed
  const hasSkills = installItems.some(i => i.type === 'skill');
  const hasSubagents = installItems.some(i => i.type === 'subagent');
  const hasCommands = installItems.some(i => i.type === 'command');

  // If only one type, no need to ask
  const types = [hasSkills, hasSubagents, hasCommands].filter(Boolean);
  if (types.length <= 1) {
    if (hasSkills) return ['skills'];
    if (hasSubagents) return ['subagents'];
    if (hasCommands) return ['commands'];
    return ['skills'];
  }

  // For skills, ask where to install in Claude
  if (hasSkills) {
    const response = await prompts({
      type: 'multiselect',
      name: 'targets',
      message: 'Where to install skills for Claude?',
      choices: CLAUDE_TARGETS.map(t => ({
        title: `${t.name} (.claude/${t.folder}/)`,
        value: t.id,
        selected: t.id === 'skills'
      })),
      min: 1,
      hint: '- Skills can also be installed as subagents or commands'
    });

    return response.targets || ['skills'];
  }

  return ['skills'];
}

async function installItems_(items, agents, claudeTargets, repoPath, manifest, cwd) {
  const installed = [];

  for (const item of items) {
    const sourcePath = path.join(repoPath, item.path);

    if (!await fs.pathExists(sourcePath)) {
      console.warn(`⚠️  Path not found: ${item.path}`);
      continue;
    }

    for (const agentId of agents) {
      const agent = AGENTS.find(a => a.id === agentId);

      if (agentId === 'claude') {
        // For Claude, install to selected targets
        const targets = getTargetsForItem(item, claudeTargets);

        for (const targetId of targets) {
          const target = CLAUDE_TARGETS.find(t => t.id === targetId);
          const destPath = path.join(cwd, agent.folder, target.folder, item.id);

          await fs.remove(destPath);
          await fs.ensureDir(path.dirname(destPath));
          await fs.copy(sourcePath, destPath);

          installed.push({
            type: item.type,
            id: item.id,
            name: item.name,
            agent: agent.name,
            target: target.name,
            path: destPath,
            version: manifest.version
          });
        }
      } else {
        // For other agents, install to skills folder
        const destPath = path.join(cwd, agent.folder, 'skills', item.id);

        await fs.remove(destPath);
        await fs.ensureDir(path.dirname(destPath));
        await fs.copy(sourcePath, destPath);

        installed.push({
          type: item.type,
          id: item.id,
          name: item.name,
          agent: agent.name,
          target: 'skills',
          path: destPath,
          version: manifest.version
        });
      }
    }
  }

  return installed;
}

function getTargetsForItem(item, claudeTargets) {
  // Map item type to default target
  const typeToTarget = {
    skill: 'skills',
    subagent: 'subagents',
    command: 'commands'
  };

  // If item type has a specific target, use it
  if (item.type !== 'skill') {
    return [typeToTarget[item.type]];
  }

  // For skills, use selected targets
  return claudeTargets;
}

async function saveInstalledManifest(cwd, data) {
  const manifestPath = path.join(cwd, INSTALLED_MANIFEST);
  let existing = { installations: [] };

  if (await fs.pathExists(manifestPath)) {
    existing = await fs.readJson(manifestPath);
  }

  // Update or add installation record
  const sourceIndex = existing.installations.findIndex(i => i.source === data.source);
  if (sourceIndex >= 0) {
    existing.installations[sourceIndex] = data;
  } else {
    existing.installations.push(data);
  }

  await fs.writeJson(manifestPath, existing, { spaces: 2 });
}

async function updateInstalledSkills() {
  const cwd = process.cwd();
  const manifestPath = path.join(cwd, INSTALLED_MANIFEST);

  if (!await fs.pathExists(manifestPath)) {
    console.log('No installed skills found. Use "add-skill <owner/repo>" to install first.');
    process.exit(0);
  }

  const installed = await fs.readJson(manifestPath);

  if (!installed.installations?.length) {
    console.log('No installations found.');
    process.exit(0);
  }

  console.log('\n🔄 Checking for updates...\n');

  for (const installation of installed.installations) {
    const { owner, repo } = parseRepoArg(installation.source);
    const ref = installation.ref || 'main';

    console.log(`📦 ${installation.source} (current: v${installation.version})`);

    const tempDir = path.join(os.tmpdir(), `add-skill-update-${Date.now()}`);

    try {
      await downloadAndExtract(owner, repo, ref, tempDir);

      const extractedFolders = await fs.readdir(tempDir);
      const repoFolder = extractedFolders[0];
      const repoPath = path.join(tempDir, repoFolder);

      const manifest = await fs.readJson(path.join(repoPath, 'manifest.json'));

      if (manifest.version === installation.version) {
        console.log(`   ✓ Already up to date (v${manifest.version})\n`);
        continue;
      }

      console.log(`   ↑ Updating to v${manifest.version}...`);

      // Re-install all items from this source
      for (const item of installation.items) {
        const allItems = [
          ...(manifest.skills || []),
          ...(manifest.subagents || []),
          ...(manifest.commands || [])
        ];

        const sourceItem = allItems.find(i => i.id === item.id);
        if (!sourceItem) {
          console.log(`   ⚠️  Item "${item.id}" no longer exists in source`);
          continue;
        }

        const sourcePath = path.join(repoPath, sourceItem.path);
        if (await fs.pathExists(sourcePath)) {
          await fs.remove(item.path);
          await fs.ensureDir(path.dirname(item.path));
          await fs.copy(sourcePath, item.path);
          console.log(`   ✓ Updated ${item.name}`);
        }
      }

      // Update version in manifest
      installation.version = manifest.version;
      installation.installedAt = new Date().toISOString();

      console.log(`   ✅ Updated to v${manifest.version}\n`);

    } catch (error) {
      console.error(`   ❌ Failed: ${error.message}\n`);
    } finally {
      await fs.remove(tempDir).catch(() => {});
    }
  }

  // Save updated manifest
  await fs.writeJson(manifestPath, installed, { spaces: 2 });
  console.log('✅ Update complete!\n');
}

async function listInstalledSkills() {
  const cwd = process.cwd();
  const manifestPath = path.join(cwd, INSTALLED_MANIFEST);

  if (!await fs.pathExists(manifestPath)) {
    console.log('No installed skills found.');
    process.exit(0);
  }

  const installed = await fs.readJson(manifestPath);

  if (!installed.installations?.length) {
    console.log('No installations found.');
    process.exit(0);
  }

  console.log('\n📦 Installed Skills\n');
  console.log('─'.repeat(60));

  for (const installation of installed.installations) {
    console.log(`\n  Source: ${installation.source}`);
    console.log(`  Version: v${installation.version}`);
    console.log(`  Installed: ${new Date(installation.installedAt).toLocaleDateString()}`);
    console.log(`  Items:`);

    for (const item of installation.items) {
      const relativePath = path.relative(cwd, item.path);
      console.log(`    • [${item.type}] ${item.name} → ${relativePath}`);
    }
  }

  console.log('\n' + '─'.repeat(60));
  console.log('\nRun "add-skill --update" to check for updates.\n');
}

function printSummary(installed, cwd) {
  console.log('\n✅ Installation Summary\n');
  console.log('─'.repeat(60));

  // Group by agent
  const byAgent = {};
  for (const item of installed) {
    const key = `${item.agent} (${item.target})`;
    if (!byAgent[key]) byAgent[key] = [];
    byAgent[key].push(item);
  }

  for (const [agent, items] of Object.entries(byAgent)) {
    console.log(`\n  ${agent}:`);
    for (const item of items) {
      const relativePath = path.relative(cwd, item.path);
      console.log(`    • [${item.type}] ${item.name}`);
      console.log(`      📁 ${relativePath}`);
    }
  }

  console.log('\n' + '─'.repeat(60));
  console.log(`\n🎉 Installed ${installed.length} item(s) successfully!`);
  console.log('💡 Run "add-skill --update" to update in the future.\n');
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
