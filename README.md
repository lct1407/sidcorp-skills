# Sidcorp Skills

AI agent skills collection for Claude and other AI agents.

## Installation

### Quick Start

```bash
# Initialize and install skills (interactive)
npx add-skill lct1407/sidcorp-skills
```

### Global Installation

```bash
npm install -g add-skill
```

After global install, use `sc` command:

```bash
sc init
sc upgrade
sc --list
```

## Commands

| Command | Description |
|---------|-------------|
| `sc init` | Initialize and install skills interactively |
| `sc upgrade` | Update CLI and all installed skills |
| `sc <owner/repo>` | Install skills from a GitHub repository |
| `sc --list` | List installed skills |

### Options

| Option | Description |
|--------|-------------|
| `--ref <branch\|tag>` | Use specific branch or tag (default: main) |

## Examples

```bash
# Interactive setup
sc init

# Install from specific repo
sc lct1407/sidcorp-skills

# Install from specific branch
sc lct1407/sidcorp-skills --ref develop

# Update everything
sc upgrade

# List installed skills
sc --list
```

## What Gets Installed

The CLI installs skills to your project's agent folders:

- **Claude**: `.claude/skills/`, `.claude/subagents/`, `.claude/commands/`
- **Antigraphity**: `.agent/skills/`

## Creating Your Own Skills Repository

Create a `manifest.json` in your repository root:

```json
{
  "name": "my-skills",
  "version": "1.0.0",
  "skills": [
    {
      "id": "my-skill",
      "name": "My Skill",
      "description": "Description of my skill",
      "path": "skills/my-skill"
    }
  ],
  "subagents": [],
  "commands": []
}
```

## License

MIT
