import { ethers } from "hardhat";

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying ERC-8004 contracts with:", deployer.address);
  console.log("Balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)));

  // 1. Deploy AgentIdentityRegistry
  const IdentityRegistry = await ethers.getContractFactory("AgentIdentityRegistry");
  const identity = await IdentityRegistry.deploy();
  await identity.waitForDeployment();
  const identityAddress = await identity.getAddress();
  console.log("AgentIdentityRegistry deployed to:", identityAddress);

  // 2. Deploy ReputationRegistry (linked to identity)
  const ReputationRegistry = await ethers.getContractFactory("ReputationRegistry");
  const reputation = await ReputationRegistry.deploy(identityAddress);
  await reputation.waitForDeployment();
  console.log("ReputationRegistry deployed to:", await reputation.getAddress());

  // 3. Deploy ValidationRegistry (linked to identity)
  const ValidationRegistry = await ethers.getContractFactory("ValidationRegistry");
  const validation = await ValidationRegistry.deploy(identityAddress);
  await validation.waitForDeployment();
  console.log("ValidationRegistry deployed to:", await validation.getAddress());

  console.log("\nDeployment complete. Set these environment variables:");
  console.log(`  ERC8004_IDENTITY_REGISTRY=${identityAddress}`);
  console.log(`  ERC8004_REPUTATION_REGISTRY=${await reputation.getAddress()}`);
  console.log(`  ERC8004_VALIDATION_REGISTRY=${await validation.getAddress()}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
