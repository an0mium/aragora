// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ReputationRegistry
 * @notice ERC-8004 Reputation Registry — on-chain calibration scores for AI agents.
 *
 * Stores feedback (Brier scores, ELO deltas, domain-specific calibration)
 * from clients who have interacted with registered agents. Feedback is
 * tagged by domain (e.g. "financial", "medical") and endpoint, enabling
 * domain-specific reputation queries.
 *
 * Linked to AgentIdentityRegistry — only registered agents can receive feedback.
 */
contract ReputationRegistry {

    // ── Types ──

    struct Feedback {
        int128 value;
        uint8 valueDecimals;
        string tag1;
        string tag2;
        string endpoint;
        string feedbackURI;
        bytes32 feedbackHash;
        bool isRevoked;
        uint256 timestamp;
    }

    // ── State ──

    address public immutable identityRegistry;

    // agentId => clientAddress => feedbackIndex => Feedback
    mapping(uint256 => mapping(address => mapping(uint64 => Feedback))) private _feedback;

    // agentId => clientAddress => next feedback index
    mapping(uint256 => mapping(address => uint64)) private _nextIndex;

    // agentId => list of client addresses that have given feedback
    mapping(uint256 => address[]) private _clients;
    mapping(uint256 => mapping(address => bool)) private _isClient;

    // ── Events ──

    event NewFeedback(
        uint256 indexed agentId,
        address indexed clientAddress,
        uint64 feedbackIndex,
        int128 value,
        uint8 valueDecimals,
        string tag1,
        string tag2,
        bytes32 feedbackHash
    );

    event FeedbackRevoked(
        uint256 indexed agentId,
        address indexed clientAddress,
        uint64 indexed feedbackIndex
    );

    event ResponseAppended(
        uint256 indexed agentId,
        address indexed clientAddress,
        uint64 feedbackIndex,
        address indexed responder,
        string responseURI,
        bytes32 responseHash
    );

    // ── Constructor ──

    /**
     * @param _identityRegistry Address of the AgentIdentityRegistry contract.
     */
    constructor(address _identityRegistry) {
        require(_identityRegistry != address(0), "Zero identity registry");
        identityRegistry = _identityRegistry;
    }

    // ── Feedback ──

    /**
     * @notice Submit feedback for an agent.
     * @param agentId The agent token ID (from IdentityRegistry).
     * @param value Signed feedback value (e.g. Brier score * 10^decimals).
     * @param valueDecimals Decimal precision of the value.
     * @param tag1 Primary domain tag (e.g. "financial", "medical").
     * @param tag2 Secondary tag (e.g. "risk_assessment").
     * @param endpoint The API endpoint or capability being rated.
     * @param feedbackURI Off-chain URI with detailed feedback.
     * @param feedbackHash SHA-256 hash of the feedback content.
     */
    function giveFeedback(
        uint256 agentId,
        int128 value,
        uint8 valueDecimals,
        string calldata tag1,
        string calldata tag2,
        string calldata endpoint,
        string calldata feedbackURI,
        bytes32 feedbackHash
    ) external {
        uint64 idx = _nextIndex[agentId][msg.sender]++;

        _feedback[agentId][msg.sender][idx] = Feedback({
            value: value,
            valueDecimals: valueDecimals,
            tag1: tag1,
            tag2: tag2,
            endpoint: endpoint,
            feedbackURI: feedbackURI,
            feedbackHash: feedbackHash,
            isRevoked: false,
            timestamp: block.timestamp
        });

        if (!_isClient[agentId][msg.sender]) {
            _clients[agentId].push(msg.sender);
            _isClient[agentId][msg.sender] = true;
        }

        emit NewFeedback(
            agentId, msg.sender, idx,
            value, valueDecimals,
            tag1, tag2, feedbackHash
        );
    }

    /**
     * @notice Revoke previously submitted feedback.
     * @dev Only the original submitter can revoke.
     */
    function revokeFeedback(uint256 agentId, uint64 feedbackIndex) external {
        Feedback storage fb = _feedback[agentId][msg.sender][feedbackIndex];
        require(fb.timestamp > 0, "Feedback not found");
        require(!fb.isRevoked, "Already revoked");
        fb.isRevoked = true;

        emit FeedbackRevoked(agentId, msg.sender, feedbackIndex);
    }

    /**
     * @notice Append a response to existing feedback (e.g. agent owner's rebuttal).
     * @dev The agent owner or the original client can append responses.
     */
    function appendResponse(
        uint256 agentId,
        address clientAddress,
        uint64 feedbackIndex,
        string calldata responseURI,
        bytes32 responseHash
    ) external {
        Feedback storage fb = _feedback[agentId][clientAddress][feedbackIndex];
        require(fb.timestamp > 0, "Feedback not found");

        emit ResponseAppended(
            agentId, clientAddress, feedbackIndex,
            msg.sender, responseURI, responseHash
        );
    }

    // ── Queries ──

    /**
     * @notice Read a specific feedback entry.
     */
    function readFeedback(
        uint256 agentId,
        address clientAddress,
        uint64 feedbackIndex
    ) external view returns (
        int128 value,
        uint8 valueDecimals,
        string memory tag1,
        string memory tag2,
        bool isRevoked
    ) {
        Feedback storage fb = _feedback[agentId][clientAddress][feedbackIndex];
        return (fb.value, fb.valueDecimals, fb.tag1, fb.tag2, fb.isRevoked);
    }

    /**
     * @notice Get aggregated reputation summary for an agent.
     * @param agentId The agent token ID.
     * @param clientAddresses Filter by specific clients (empty = all clients).
     * @param tag1 Filter by primary domain tag (empty = all).
     * @param tag2 Filter by secondary tag (empty = all).
     * @return count Number of matching (non-revoked) feedback entries.
     * @return summaryValue Sum of values (divide by count for average).
     * @return summaryValueDecimals Decimal precision of the summary.
     */
    function getSummary(
        uint256 agentId,
        address[] calldata clientAddresses,
        string calldata tag1,
        string calldata tag2
    ) external view returns (
        uint64 count,
        int128 summaryValue,
        uint8 summaryValueDecimals
    ) {
        // Copy client list to memory (can't mix calldata and storage in ternary)
        address[] memory clients;
        if (clientAddresses.length > 0) {
            clients = new address[](clientAddresses.length);
            for (uint256 k = 0; k < clientAddresses.length; k++) {
                clients[k] = clientAddresses[k];
            }
        } else {
            clients = _clients[agentId];
        }

        bytes32 t1Hash = bytes(tag1).length > 0 ? keccak256(bytes(tag1)) : bytes32(0);
        bytes32 t2Hash = bytes(tag2).length > 0 ? keccak256(bytes(tag2)) : bytes32(0);

        for (uint256 i = 0; i < clients.length; i++) {
            uint64 maxIdx = _nextIndex[agentId][clients[i]];
            for (uint64 j = 0; j < maxIdx; j++) {
                Feedback storage fb = _feedback[agentId][clients[i]][j];
                if (fb.isRevoked) continue;
                if (t1Hash != bytes32(0) && keccak256(bytes(fb.tag1)) != t1Hash) continue;
                if (t2Hash != bytes32(0) && keccak256(bytes(fb.tag2)) != t2Hash) continue;

                summaryValue += fb.value;
                if (summaryValueDecimals == 0) {
                    summaryValueDecimals = fb.valueDecimals;
                }
                count++;
            }
        }
    }

    /**
     * @notice Get all client addresses that have given feedback to an agent.
     */
    function getClients(uint256 agentId) external view returns (address[] memory) {
        return _clients[agentId];
    }

    /**
     * @notice Get the last feedback index for a client-agent pair.
     */
    function getLastIndex(
        uint256 agentId,
        address clientAddress
    ) external view returns (uint64) {
        return _nextIndex[agentId][clientAddress];
    }

    /**
     * @notice Get the linked identity registry address.
     */
    function getIdentityRegistry() external view returns (address) {
        return identityRegistry;
    }
}
