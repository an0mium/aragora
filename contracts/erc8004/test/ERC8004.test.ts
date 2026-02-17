import { expect } from "chai";
import { ethers } from "hardhat";
import { loadFixture } from "@nomicfoundation/hardhat-toolbox/network-helpers";

describe("ERC-8004 Contracts", function () {
  async function deployFixture() {
    const [owner, client, validator, other] = await ethers.getSigners();

    const IdentityRegistry = await ethers.getContractFactory("AgentIdentityRegistry");
    const identity = await IdentityRegistry.deploy();

    const identityAddress = await identity.getAddress();

    const ReputationRegistry = await ethers.getContractFactory("ReputationRegistry");
    const reputation = await ReputationRegistry.deploy(identityAddress);

    const ValidationRegistry = await ethers.getContractFactory("ValidationRegistry");
    const validation = await ValidationRegistry.deploy(identityAddress);

    return { identity, reputation, validation, owner, client, validator, other };
  }

  // ── AgentIdentityRegistry ──

  describe("AgentIdentityRegistry", function () {
    it("should register an agent and mint ERC-721 token", async function () {
      const { identity, owner } = await loadFixture(deployFixture);
      const tx = await identity.register("ipfs://agent-metadata", []);
      const receipt = await tx.wait();

      expect(await identity.ownerOf(0)).to.equal(owner.address);
      expect(await identity.tokenURI(0)).to.equal("ipfs://agent-metadata");
    });

    it("should register with initial metadata", async function () {
      const { identity } = await loadFixture(deployFixture);
      const metadata = [
        { metadataKey: "model", metadataValue: ethers.toUtf8Bytes("claude-3.5") },
        { metadataKey: "domain", metadataValue: ethers.toUtf8Bytes("financial") },
      ];
      await identity.register("ipfs://agent", metadata);

      const model = await identity.getMetadata(0, "model");
      expect(ethers.toUtf8String(model)).to.equal("claude-3.5");

      const domain = await identity.getMetadata(0, "domain");
      expect(ethers.toUtf8String(domain)).to.equal("financial");
    });

    it("should update agent URI", async function () {
      const { identity } = await loadFixture(deployFixture);
      await identity.register("ipfs://old", []);
      await identity.setAgentURI(0, "ipfs://new");
      expect(await identity.tokenURI(0)).to.equal("ipfs://new");
    });

    it("should reject URI update from non-owner", async function () {
      const { identity, other } = await loadFixture(deployFixture);
      await identity.register("ipfs://agent", []);
      await expect(
        identity.connect(other).setAgentURI(0, "ipfs://hacked")
      ).to.be.revertedWith("Not agent owner");
    });

    it("should set and get metadata", async function () {
      const { identity } = await loadFixture(deployFixture);
      await identity.register("ipfs://agent", []);
      await identity.setMetadata(0, "calibration", ethers.toUtf8Bytes("0.85"));

      const val = await identity.getMetadata(0, "calibration");
      expect(ethers.toUtf8String(val)).to.equal("0.85");
    });

    it("should set agent wallet with valid signature", async function () {
      const { identity, other } = await loadFixture(deployFixture);
      await identity.register("ipfs://agent", []);

      const agentId = 0;
      const deadline = Math.floor(Date.now() / 1000) + 3600;
      const chainId = (await ethers.provider.getNetwork()).chainId;

      const hash = ethers.solidityPackedKeccak256(
        ["uint256", "address", "uint256", "uint256"],
        [agentId, other.address, deadline, chainId]
      );
      const signature = await other.signMessage(ethers.getBytes(hash));

      await identity.setAgentWallet(agentId, other.address, deadline, signature);
      expect(await identity.getAgentWallet(agentId)).to.equal(other.address);
    });

    it("should unset agent wallet", async function () {
      const { identity, other } = await loadFixture(deployFixture);
      await identity.register("ipfs://agent", []);

      const agentId = 0;
      const deadline = Math.floor(Date.now() / 1000) + 3600;
      const chainId = (await ethers.provider.getNetwork()).chainId;
      const hash = ethers.solidityPackedKeccak256(
        ["uint256", "address", "uint256", "uint256"],
        [agentId, other.address, deadline, chainId]
      );
      const signature = await other.signMessage(ethers.getBytes(hash));
      await identity.setAgentWallet(agentId, other.address, deadline, signature);

      await identity.unsetAgentWallet(agentId);
      expect(await identity.getAgentWallet(agentId)).to.equal(ethers.ZeroAddress);
    });

    it("should enumerate tokens via ERC721Enumerable", async function () {
      const { identity } = await loadFixture(deployFixture);
      await identity.register("ipfs://a1", []);
      await identity.register("ipfs://a2", []);

      expect(await identity.totalSupply()).to.equal(2);
      expect(await identity.tokenByIndex(0)).to.equal(0);
      expect(await identity.tokenByIndex(1)).to.equal(1);
    });

    it("should emit Registered event", async function () {
      const { identity, owner } = await loadFixture(deployFixture);
      await expect(identity.register("ipfs://agent", []))
        .to.emit(identity, "Registered")
        .withArgs(0, "ipfs://agent", owner.address);
    });
  });

  // ── ReputationRegistry ──

  describe("ReputationRegistry", function () {
    it("should accept feedback from any address", async function () {
      const { reputation, client } = await loadFixture(deployFixture);

      await reputation.connect(client).giveFeedback(
        0,        // agentId
        850,      // value (Brier score * 1000)
        3,        // decimals
        "financial",
        "risk_assessment",
        "/api/v1/analyze",
        "ipfs://feedback-1",
        ethers.id("feedback-content")
      );

      const [value, decimals, tag1, tag2, revoked] =
        await reputation.readFeedback(0, client.address, 0);
      expect(value).to.equal(850);
      expect(decimals).to.equal(3);
      expect(tag1).to.equal("financial");
      expect(tag2).to.equal("risk_assessment");
      expect(revoked).to.be.false;
    });

    it("should track multiple feedback entries", async function () {
      const { reputation, client } = await loadFixture(deployFixture);

      await reputation.connect(client).giveFeedback(
        0, 900, 3, "medical", "", "/api/v1/diagnose", "", ethers.ZeroHash
      );
      await reputation.connect(client).giveFeedback(
        0, 750, 3, "financial", "", "/api/v1/analyze", "", ethers.ZeroHash
      );

      expect(await reputation.getLastIndex(0, client.address)).to.equal(2);
    });

    it("should revoke feedback", async function () {
      const { reputation, client } = await loadFixture(deployFixture);

      await reputation.connect(client).giveFeedback(
        0, 500, 3, "test", "", "", "", ethers.ZeroHash
      );
      await reputation.connect(client).revokeFeedback(0, 0);

      const [, , , , revoked] = await reputation.readFeedback(0, client.address, 0);
      expect(revoked).to.be.true;
    });

    it("should reject revocation from non-submitter", async function () {
      const { reputation, client, other } = await loadFixture(deployFixture);

      await reputation.connect(client).giveFeedback(
        0, 500, 3, "test", "", "", "", ethers.ZeroHash
      );

      // other tries to revoke client's feedback - fails because feedback[0][other][0] doesn't exist
      await expect(
        reputation.connect(other).revokeFeedback(0, 0)
      ).to.be.revertedWith("Feedback not found");
    });

    it("should compute summary across clients", async function () {
      const { reputation, client, validator } = await loadFixture(deployFixture);

      // Two clients give feedback
      await reputation.connect(client).giveFeedback(
        0, 800, 3, "financial", "", "", "", ethers.ZeroHash
      );
      await reputation.connect(validator).giveFeedback(
        0, 900, 3, "financial", "", "", "", ethers.ZeroHash
      );

      const [count, summaryValue, decimals] = await reputation.getSummary(0, [], "", "");
      expect(count).to.equal(2);
      expect(summaryValue).to.equal(1700); // 800 + 900
    });

    it("should filter summary by tag", async function () {
      const { reputation, client } = await loadFixture(deployFixture);

      await reputation.connect(client).giveFeedback(
        0, 800, 3, "financial", "", "", "", ethers.ZeroHash
      );
      await reputation.connect(client).giveFeedback(
        0, 900, 3, "medical", "", "", "", ethers.ZeroHash
      );

      const [count] = await reputation.getSummary(0, [], "medical", "");
      expect(count).to.equal(1);
    });

    it("should exclude revoked feedback from summary", async function () {
      const { reputation, client } = await loadFixture(deployFixture);

      await reputation.connect(client).giveFeedback(
        0, 800, 3, "test", "", "", "", ethers.ZeroHash
      );
      await reputation.connect(client).giveFeedback(
        0, 200, 3, "test", "", "", "", ethers.ZeroHash
      );
      await reputation.connect(client).revokeFeedback(0, 1);

      const [count, summaryValue] = await reputation.getSummary(0, [], "", "");
      expect(count).to.equal(1);
      expect(summaryValue).to.equal(800);
    });

    it("should track client list", async function () {
      const { reputation, client, validator } = await loadFixture(deployFixture);

      await reputation.connect(client).giveFeedback(
        0, 100, 0, "", "", "", "", ethers.ZeroHash
      );
      await reputation.connect(validator).giveFeedback(
        0, 200, 0, "", "", "", "", ethers.ZeroHash
      );

      const clients = await reputation.getClients(0);
      expect(clients).to.have.lengthOf(2);
      expect(clients).to.include(client.address);
      expect(clients).to.include(validator.address);
    });

    it("should emit NewFeedback event", async function () {
      const { reputation, client } = await loadFixture(deployFixture);

      await expect(
        reputation.connect(client).giveFeedback(
          0, 850, 3, "financial", "risk", "/api", "ipfs://fb", ethers.id("content")
        )
      ).to.emit(reputation, "NewFeedback");
    });

    it("should link to identity registry", async function () {
      const { reputation, identity } = await loadFixture(deployFixture);
      expect(await reputation.getIdentityRegistry()).to.equal(await identity.getAddress());
    });
  });

  // ── ValidationRegistry ──

  describe("ValidationRegistry", function () {
    it("should submit a validation request", async function () {
      const { validation, validator, owner } = await loadFixture(deployFixture);

      const requestHash = ethers.id("validation-request-1");
      await validation.validationRequest(
        validator.address, 0, "ipfs://request", requestHash
      );

      const [addr, agentId, response, , , lastUpdate] =
        await validation.getValidationStatus(requestHash);
      expect(addr).to.equal(validator.address);
      expect(agentId).to.equal(0);
      expect(response).to.equal(0); // PENDING
      expect(lastUpdate).to.be.gt(0);
    });

    it("should reject duplicate request hash", async function () {
      const { validation, validator } = await loadFixture(deployFixture);

      const requestHash = ethers.id("duplicate");
      await validation.validationRequest(validator.address, 0, "", requestHash);

      await expect(
        validation.validationRequest(validator.address, 0, "", requestHash)
      ).to.be.revertedWith("Request hash already used");
    });

    it("should accept response from designated validator", async function () {
      const { validation, validator } = await loadFixture(deployFixture);

      const requestHash = ethers.id("val-req");
      await validation.validationRequest(validator.address, 0, "", requestHash);

      await validation.connect(validator).validationResponse(
        requestHash,
        1, // PASS
        "ipfs://response",
        ethers.id("response-content"),
        "safety"
      );

      const [, , response, responseHash, tag] =
        await validation.getValidationStatus(requestHash);
      expect(response).to.equal(1); // PASS
      expect(tag).to.equal("safety");
    });

    it("should reject response from non-validator", async function () {
      const { validation, validator, other } = await loadFixture(deployFixture);

      const requestHash = ethers.id("val-req-2");
      await validation.validationRequest(validator.address, 0, "", requestHash);

      await expect(
        validation.connect(other).validationResponse(
          requestHash, 1, "", ethers.ZeroHash, ""
        )
      ).to.be.revertedWith("Not designated validator");
    });

    it("should reject invalid response codes", async function () {
      const { validation, validator } = await loadFixture(deployFixture);

      const requestHash = ethers.id("val-req-3");
      await validation.validationRequest(validator.address, 0, "", requestHash);

      await expect(
        validation.connect(validator).validationResponse(
          requestHash, 0, "", ethers.ZeroHash, "" // 0 = PENDING, invalid
        )
      ).to.be.revertedWith("Invalid response code");

      await expect(
        validation.connect(validator).validationResponse(
          requestHash, 4, "", ethers.ZeroHash, "" // 4 = out of range
        )
      ).to.be.revertedWith("Invalid response code");
    });

    it("should compute summary with tag filtering", async function () {
      const { validation, validator, other } = await loadFixture(deployFixture);

      // Two validation requests for the same agent
      const h1 = ethers.id("req-1");
      const h2 = ethers.id("req-2");
      await validation.validationRequest(validator.address, 0, "", h1);
      await validation.validationRequest(other.address, 0, "", h2);

      await validation.connect(validator).validationResponse(
        h1, 1, "", ethers.ZeroHash, "safety"  // PASS
      );
      await validation.connect(other).validationResponse(
        h2, 2, "", ethers.ZeroHash, "accuracy" // FAIL
      );

      // All validations
      const [countAll] = await validation.getSummary(0, [], "");
      expect(countAll).to.equal(2);

      // Filter by tag
      const [countSafety] = await validation.getSummary(0, [], "safety");
      expect(countSafety).to.equal(1);
    });

    it("should track agent validations", async function () {
      const { validation, validator } = await loadFixture(deployFixture);

      const h1 = ethers.id("a-1");
      const h2 = ethers.id("a-2");
      await validation.validationRequest(validator.address, 0, "", h1);
      await validation.validationRequest(validator.address, 0, "", h2);

      const hashes = await validation.getAgentValidations(0);
      expect(hashes).to.have.lengthOf(2);
    });

    it("should track validator requests", async function () {
      const { validation, validator } = await loadFixture(deployFixture);

      const h1 = ethers.id("v-1");
      const h2 = ethers.id("v-2");
      await validation.validationRequest(validator.address, 0, "", h1);
      await validation.validationRequest(validator.address, 1, "", h2);

      const hashes = await validation.getValidatorRequests(validator.address);
      expect(hashes).to.have.lengthOf(2);
    });

    it("should emit ValidationRequested event", async function () {
      const { validation, validator } = await loadFixture(deployFixture);
      const requestHash = ethers.id("ev-1");

      await expect(
        validation.validationRequest(validator.address, 0, "ipfs://req", requestHash)
      ).to.emit(validation, "ValidationRequested")
        .withArgs(validator.address, 0, "ipfs://req", requestHash);
    });

    it("should emit ValidationResponded event", async function () {
      const { validation, validator } = await loadFixture(deployFixture);
      const requestHash = ethers.id("ev-2");

      await validation.validationRequest(validator.address, 0, "", requestHash);

      await expect(
        validation.connect(validator).validationResponse(
          requestHash, 1, "ipfs://resp", ethers.id("resp"), "safety"
        )
      ).to.emit(validation, "ValidationResponded");
    });

    it("should link to identity registry", async function () {
      const { validation, identity } = await loadFixture(deployFixture);
      expect(await validation.getIdentityRegistry()).to.equal(await identity.getAddress());
    });
  });

  // ── Cross-Registry Integration ──

  describe("Cross-Registry Integration", function () {
    it("should register agent then submit feedback and validation", async function () {
      const { identity, reputation, validation, owner, client, validator } =
        await loadFixture(deployFixture);

      // Register an agent
      await identity.register("ipfs://agent-1", [
        { metadataKey: "model", metadataValue: ethers.toUtf8Bytes("claude-4") },
      ]);

      // Submit reputation feedback
      await reputation.connect(client).giveFeedback(
        0, 920, 3, "financial", "analysis", "/api/v1/analyze",
        "ipfs://fb-1", ethers.id("feedback-body")
      );

      // Submit validation request and response
      const reqHash = ethers.id("cross-reg-val");
      await validation.validationRequest(validator.address, 0, "ipfs://val-req", reqHash);
      await validation.connect(validator).validationResponse(
        reqHash, 1, "ipfs://val-resp", ethers.id("val-body"), "safety"
      );

      // Verify all data is consistent
      expect(await identity.ownerOf(0)).to.equal(owner.address);

      const [fbValue] = await reputation.readFeedback(0, client.address, 0);
      expect(fbValue).to.equal(920);

      const [, , valResponse] = await validation.getValidationStatus(reqHash);
      expect(valResponse).to.equal(1); // PASS
    });
  });
});
