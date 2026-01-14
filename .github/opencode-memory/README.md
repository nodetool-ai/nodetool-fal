# OpenCode Memory

This directory contains context and instructions for the OpenCode autonomous agent that maintains this repository.

## Purpose

The agent periodically scans FAL's model catalog to discover new endpoints and adds corresponding NodeTool nodes.

## Files

- `README.md` - This file
- `repository-context.md` - Overview of the repository structure
- `node-creation-guide.md` - Step-by-step instructions for creating new nodes
- `features.md` - Log of features added by the agent

## Related Workflows

- `.github/workflows/opencode-fal-model-sync.yml` - Scheduled workflow that runs the agent
