#!/usr/bin/env python3
"""
Generate TypeScript SDK from OpenAPI specification.

This script uses OpenAPI Generator to create a fully-typed TypeScript/JavaScript
SDK from the Aragora OpenAPI specification.

Prerequisites:
    npm install -g @openapitools/openapi-generator-cli

    Or use Docker:
    docker pull openapitools/openapi-generator-cli

Usage:
    python scripts/generate_sdk.py
    python scripts/generate_sdk.py --output ./sdk-output
    python scripts/generate_sdk.py --generator typescript-fetch
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# SDK Configuration
SDK_CONFIG = {
    "npmName": "@aragora/client",
    "npmVersion": "2.0.0",
    "supportsES6": True,
    "withInterfaces": True,
    "typescriptThreePlus": True,
    "enumPropertyNaming": "UPPERCASE",
    "modelPropertyNaming": "camelCase",
    "paramNaming": "camelCase",
}


def check_prerequisites() -> tuple[bool, str]:
    """Check if OpenAPI Generator is available."""
    # Check for npx (npm)
    if shutil.which("npx"):
        result = subprocess.run(
            ["npx", "@openapitools/openapi-generator-cli", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "npx"

    # Check for Docker
    if shutil.which("docker"):
        result = subprocess.run(
            ["docker", "images", "-q", "openapitools/openapi-generator-cli"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            return True, "docker"

    return False, ""


def generate_sdk_npm(
    openapi_path: Path,
    output_dir: Path,
    generator: str,
    config_file: Path,
) -> int:
    """Generate SDK using npm/npx."""
    cmd = [
        "npx",
        "@openapitools/openapi-generator-cli",
        "generate",
        "-i",
        str(openapi_path),
        "-g",
        generator,
        "-o",
        str(output_dir),
        "-c",
        str(config_file),
        "--additional-properties=supportsES6=true,withInterfaces=true",
    ]

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def generate_sdk_docker(
    openapi_path: Path,
    output_dir: Path,
    generator: str,
    config_file: Path,
) -> int:
    """Generate SDK using Docker."""
    openapi_abs = openapi_path.resolve()
    output_abs = output_dir.resolve()
    config_abs = config_file.resolve()
    project_root = openapi_abs.parent.parent

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{project_root}:/local",
        "openapitools/openapi-generator-cli",
        "generate",
        "-i",
        f"/local/{openapi_abs.relative_to(project_root)}",
        "-g",
        generator,
        "-o",
        f"/local/{output_abs.relative_to(project_root)}",
        "-c",
        f"/local/{config_abs.relative_to(project_root)}",
    ]

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def create_package_json(output_dir: Path) -> None:
    """Create package.json for the SDK."""
    package = {
        "name": "@aragora/client",
        "version": "2.0.0",
        "description": "TypeScript/JavaScript client for Aragora Multi-Agent Debate API",
        "main": "dist/index.js",
        "module": "dist/index.mjs",
        "types": "dist/index.d.ts",
        "scripts": {
            "build": "tsc",
            "test": "jest",
            "lint": "eslint src/",
            "prepublishOnly": "npm run build",
        },
        "keywords": ["aragora", "multi-agent", "debate", "ai", "llm", "consensus"],
        "author": "Aragora Team",
        "license": "MIT",
        "dependencies": {"cross-fetch": "^4.0.0"},
        "devDependencies": {"@types/node": "^20.0.0", "typescript": "^5.0.0"},
        "engines": {"node": ">=18.0.0"},
    }
    with open(output_dir / "package.json", "w") as f:
        json.dump(package, f, indent=2)


def create_readme(output_dir: Path) -> None:
    """Create README.md for the SDK."""
    readme = """# @aragora/client

TypeScript/JavaScript client for the Aragora Multi-Agent Debate API.

## Installation

```bash
npm install @aragora/client
```

## Quick Start

```typescript
import { AragoraClient, Configuration } from '@aragora/client';

const config = new Configuration({
  basePath: 'https://api.aragora.ai',
  accessToken: 'your-api-token',
});

const client = new AragoraClient(config);

// Start a debate
const debate = await client.createDebate({
  task: 'What is the best programming language for beginners?',
  agents: ['claude', 'gpt4', 'gemini'],
  protocol: { rounds: 3, consensusMode: 'majority' },
});

// Get debate status
const status = await client.getDebate(debate.id);

// List agents
const agents = await client.listAgents();
```

## API Methods

- `createDebate(request)` - Start a new debate
- `getDebate(id)` - Get debate by ID
- `listDebates(options?)` - List all debates
- `listAgents()` - List available agents
- `getAgentRankings()` - Get agent ELO rankings
- `getConsensusMemory()` - Get consensus memory
- `healthCheck()` - Check API health

## License

MIT
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate TypeScript SDK from OpenAPI spec")
    parser.add_argument("--openapi", type=Path, default=Path("docs/api/openapi.json"))
    parser.add_argument("--output", type=Path, default=Path("sdk/typescript"))
    parser.add_argument(
        "--generator",
        default="typescript-fetch",
        choices=["typescript-fetch", "typescript-axios", "typescript-node"],
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    openapi_path = project_root / args.openapi
    output_dir = project_root / args.output

    if not openapi_path.exists():
        print(f"Error: OpenAPI spec not found at {openapi_path}")
        return 1

    available, method = check_prerequisites()
    if not available:
        print("Error: OpenAPI Generator not found.")
        print("Install with: npm install -g @openapitools/openapi-generator-cli")
        return 1

    print(f"Using OpenAPI Generator via {method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_file = output_dir / "openapi-generator-config.json"
    with open(config_file, "w") as f:
        json.dump(SDK_CONFIG, f, indent=2)

    if method == "npx":
        result = generate_sdk_npm(openapi_path, output_dir, args.generator, config_file)
    else:
        result = generate_sdk_docker(openapi_path, output_dir, args.generator, config_file)

    if result != 0:
        print("Error: SDK generation failed")
        return result

    create_package_json(output_dir)
    create_readme(output_dir)

    print(f"\nSDK generated successfully at {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
