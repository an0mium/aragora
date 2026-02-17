// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/**
 * @title AgentIdentityRegistry
 * @notice ERC-8004 Agent Identity Registry — ERC-721 tokens representing AI agents.
 *
 * Each token represents a registered AI agent with:
 * - A URI pointing to the agent's metadata (model name, capabilities, etc.)
 * - Key-value metadata stored on-chain (calibration scores, domain tags)
 * - An optional wallet address for the agent to sign transactions
 *
 * This is the canonical on-chain registry for the Aragora Decision Integrity
 * Platform. Agents registered here can accumulate reputation via the
 * ReputationRegistry and undergo validation via the ValidationRegistry.
 */
contract AgentIdentityRegistry is ERC721, ERC721Enumerable, Ownable {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    // ── Types ──

    struct MetadataEntry {
        string metadataKey;
        bytes metadataValue;
    }

    // ── State ──

    uint256 private _nextTokenId;

    // tokenId => agent URI (off-chain metadata pointer)
    mapping(uint256 => string) private _agentURIs;

    // tokenId => metadataKey => metadataValue
    mapping(uint256 => mapping(string => bytes)) private _metadata;

    // tokenId => wallet address (optional, for agent-signed transactions)
    mapping(uint256 => address) private _agentWallets;

    // ── Events ──

    event Registered(
        uint256 indexed agentId,
        string agentURI,
        address indexed owner
    );

    event URIUpdated(
        uint256 indexed agentId,
        string newURI,
        address indexed updatedBy
    );

    event MetadataSet(
        uint256 indexed agentId,
        string indexed indexedMetadataKey,
        string metadataKey,
        bytes metadataValue
    );

    event WalletSet(uint256 indexed agentId, address wallet);
    event WalletUnset(uint256 indexed agentId);

    // ── Constructor ──

    constructor() ERC721("Aragora Agent", "AGENT") Ownable(msg.sender) {}

    // ── Registration ──

    /**
     * @notice Register a new AI agent as an ERC-721 token.
     * @param agentURI URI pointing to agent metadata (IPFS, HTTPS, etc.)
     * @param metadata Initial key-value metadata entries to store on-chain.
     * @return agentId The token ID of the newly registered agent.
     */
    function register(
        string calldata agentURI,
        MetadataEntry[] calldata metadata
    ) external returns (uint256 agentId) {
        agentId = _nextTokenId++;
        _safeMint(msg.sender, agentId);
        _agentURIs[agentId] = agentURI;

        for (uint256 i = 0; i < metadata.length; i++) {
            _metadata[agentId][metadata[i].metadataKey] = metadata[i].metadataValue;
            emit MetadataSet(
                agentId,
                metadata[i].metadataKey,
                metadata[i].metadataKey,
                metadata[i].metadataValue
            );
        }

        emit Registered(agentId, agentURI, msg.sender);
    }

    // ── URI Management ──

    /**
     * @notice Update the metadata URI for an agent.
     * @dev Only the token owner can update the URI.
     */
    function setAgentURI(uint256 agentId, string calldata newURI) external {
        require(ownerOf(agentId) == msg.sender, "Not agent owner");
        _agentURIs[agentId] = newURI;
        emit URIUpdated(agentId, newURI, msg.sender);
    }

    /**
     * @notice Returns the URI for a given token (ERC-721 standard).
     */
    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        _requireOwned(tokenId);
        return _agentURIs[tokenId];
    }

    // ── Metadata ──

    /**
     * @notice Get an on-chain metadata value for an agent.
     * @param agentId The agent token ID.
     * @param metadataKey The key to look up.
     * @return The raw bytes value (decode off-chain).
     */
    function getMetadata(
        uint256 agentId,
        string calldata metadataKey
    ) external view returns (bytes memory) {
        _requireOwned(agentId);
        return _metadata[agentId][metadataKey];
    }

    /**
     * @notice Set an on-chain metadata value for an agent.
     * @dev Only the token owner can set metadata.
     */
    function setMetadata(
        uint256 agentId,
        string calldata metadataKey,
        bytes calldata metadataValue
    ) external {
        require(ownerOf(agentId) == msg.sender, "Not agent owner");
        _metadata[agentId][metadataKey] = metadataValue;
        emit MetadataSet(agentId, metadataKey, metadataKey, metadataValue);
    }

    // ── Wallet Management ──

    /**
     * @notice Assign a wallet address to an agent using EIP-712 signature.
     * @dev The new wallet must sign a message proving ownership.
     * @param agentId The agent token ID.
     * @param newWallet The wallet address to assign.
     * @param deadline Signature expiry timestamp.
     * @param signature EIP-712 signature from the new wallet.
     */
    function setAgentWallet(
        uint256 agentId,
        address newWallet,
        uint256 deadline,
        bytes calldata signature
    ) external {
        require(ownerOf(agentId) == msg.sender, "Not agent owner");
        require(block.timestamp <= deadline, "Signature expired");
        require(newWallet != address(0), "Zero address");

        // Verify the new wallet signed this assignment
        bytes32 hash = keccak256(
            abi.encodePacked(agentId, newWallet, deadline, block.chainid)
        );
        address signer = hash.toEthSignedMessageHash().recover(signature);
        require(signer == newWallet, "Invalid wallet signature");

        _agentWallets[agentId] = newWallet;
        emit WalletSet(agentId, newWallet);
    }

    /**
     * @notice Get the wallet address assigned to an agent.
     */
    function getAgentWallet(uint256 agentId) external view returns (address) {
        _requireOwned(agentId);
        return _agentWallets[agentId];
    }

    /**
     * @notice Remove the wallet assignment from an agent.
     */
    function unsetAgentWallet(uint256 agentId) external {
        require(ownerOf(agentId) == msg.sender, "Not agent owner");
        delete _agentWallets[agentId];
        emit WalletUnset(agentId);
    }

    // ── ERC-721 Overrides ──

    function _update(
        address to,
        uint256 tokenId,
        address auth
    ) internal override(ERC721, ERC721Enumerable) returns (address) {
        return super._update(to, tokenId, auth);
    }

    function _increaseBalance(
        address account,
        uint128 value
    ) internal override(ERC721, ERC721Enumerable) {
        super._increaseBalance(account, value);
    }

    function supportsInterface(
        bytes4 interfaceId
    ) public view override(ERC721, ERC721Enumerable) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}
