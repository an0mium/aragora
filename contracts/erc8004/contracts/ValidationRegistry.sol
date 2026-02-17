// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ValidationRegistry
 * @notice ERC-8004 Validation Registry — on-chain validation records for AI agents.
 *
 * Third-party validators can submit validation requests for registered agents
 * and record pass/fail/revoked responses. Supports tag-based filtering and
 * aggregated summaries for reputation assessment.
 *
 * Linked to AgentIdentityRegistry — only registered agents can be validated.
 */
contract ValidationRegistry {

    // ── Types ──

    enum Response {
        PENDING,    // 0 - No response yet
        PASS,       // 1 - Validation passed
        FAIL,       // 2 - Validation failed
        REVOKED     // 3 - Validation revoked
    }

    struct ValidationRecord {
        address validatorAddress;
        uint256 agentId;
        Response response;
        bytes32 responseHash;
        string tag;
        string requestURI;
        bytes32 requestHash;
        uint256 lastUpdate;
    }

    // ── State ──

    address public immutable identityRegistry;

    // requestHash => ValidationRecord
    mapping(bytes32 => ValidationRecord) private _records;

    // agentId => list of request hashes
    mapping(uint256 => bytes32[]) private _agentValidations;

    // validatorAddress => list of request hashes
    mapping(address => bytes32[]) private _validatorRequests;

    // ── Events ──

    event ValidationRequested(
        address indexed validatorAddress,
        uint256 indexed agentId,
        string requestURI,
        bytes32 indexed requestHash
    );

    event ValidationResponded(
        address indexed validatorAddress,
        uint256 indexed agentId,
        bytes32 indexed requestHash,
        uint8 response,
        string responseURI,
        bytes32 responseHash,
        string tag
    );

    // ── Constructor ──

    /**
     * @param _identityRegistry Address of the AgentIdentityRegistry contract.
     */
    constructor(address _identityRegistry) {
        require(_identityRegistry != address(0), "Zero identity registry");
        identityRegistry = _identityRegistry;
    }

    // ── Validation Request ──

    /**
     * @notice Submit a validation request for an agent.
     * @param validatorAddress The validator being asked to validate.
     * @param agentId The agent token ID (from IdentityRegistry).
     * @param requestURI Off-chain URI with validation request details.
     * @param requestHash SHA-256 hash of the request content (used as unique key).
     */
    function validationRequest(
        address validatorAddress,
        uint256 agentId,
        string calldata requestURI,
        bytes32 requestHash
    ) external {
        require(validatorAddress != address(0), "Zero validator address");
        require(_records[requestHash].lastUpdate == 0, "Request hash already used");

        _records[requestHash] = ValidationRecord({
            validatorAddress: validatorAddress,
            agentId: agentId,
            response: Response.PENDING,
            responseHash: bytes32(0),
            tag: "",
            requestURI: requestURI,
            requestHash: requestHash,
            lastUpdate: block.timestamp
        });

        _agentValidations[agentId].push(requestHash);
        _validatorRequests[validatorAddress].push(requestHash);

        emit ValidationRequested(validatorAddress, agentId, requestURI, requestHash);
    }

    // ── Validation Response ──

    /**
     * @notice Submit a validation response.
     * @dev Only the designated validator can respond to a request.
     * @param requestHash Hash identifying the validation request.
     * @param response Response code (1=PASS, 2=FAIL, 3=REVOKED).
     * @param responseURI URI to detailed response data.
     * @param responseHash Hash of the response content.
     * @param tag Category tag for the validation (e.g. "safety", "accuracy").
     */
    function validationResponse(
        bytes32 requestHash,
        uint8 response,
        string calldata responseURI,
        bytes32 responseHash,
        string calldata tag
    ) external {
        ValidationRecord storage record = _records[requestHash];
        require(record.lastUpdate > 0, "Request not found");
        require(record.validatorAddress == msg.sender, "Not designated validator");
        require(response >= 1 && response <= 3, "Invalid response code");

        record.response = Response(response);
        record.responseHash = responseHash;
        record.tag = tag;
        record.lastUpdate = block.timestamp;

        emit ValidationResponded(
            msg.sender, record.agentId, requestHash,
            response, responseURI, responseHash, tag
        );
    }

    // ── Queries ──

    /**
     * @notice Get the current status of a validation request.
     * @param requestHash Hash identifying the validation request.
     */
    function getValidationStatus(
        bytes32 requestHash
    ) external view returns (
        address validatorAddress,
        uint256 agentId,
        uint8 response,
        bytes32 responseHash,
        string memory tag,
        uint256 lastUpdate
    ) {
        ValidationRecord storage record = _records[requestHash];
        return (
            record.validatorAddress,
            record.agentId,
            uint8(record.response),
            record.responseHash,
            record.tag,
            record.lastUpdate
        );
    }

    /**
     * @notice Get aggregated validation summary for an agent.
     * @param agentId The agent token ID.
     * @param validatorAddresses Filter by validators (empty = all).
     * @param tag Filter by tag (empty = all).
     * @return count Number of matching completed validations.
     * @return averageResponse Rounded average response code.
     */
    function getSummary(
        uint256 agentId,
        address[] calldata validatorAddresses,
        string calldata tag
    ) external view returns (
        uint64 count,
        uint8 averageResponse
    ) {
        bytes32[] storage hashes = _agentValidations[agentId];
        bytes32 tagHash = bytes(tag).length > 0 ? keccak256(bytes(tag)) : bytes32(0);
        uint256 responseSum;

        for (uint256 i = 0; i < hashes.length; i++) {
            ValidationRecord storage record = _records[hashes[i]];

            // Skip pending validations
            if (record.response == Response.PENDING) continue;

            // Filter by tag
            if (tagHash != bytes32(0) && keccak256(bytes(record.tag)) != tagHash) continue;

            // Filter by validator addresses
            if (validatorAddresses.length > 0) {
                bool found = false;
                for (uint256 j = 0; j < validatorAddresses.length; j++) {
                    if (record.validatorAddress == validatorAddresses[j]) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue;
            }

            responseSum += uint8(record.response);
            count++;
        }

        if (count > 0) {
            averageResponse = uint8(responseSum / count);
        }
    }

    /**
     * @notice Get all validation request hashes for an agent.
     */
    function getAgentValidations(uint256 agentId) external view returns (bytes32[] memory) {
        return _agentValidations[agentId];
    }

    /**
     * @notice Get all validation request hashes assigned to a validator.
     */
    function getValidatorRequests(address validatorAddress) external view returns (bytes32[] memory) {
        return _validatorRequests[validatorAddress];
    }

    /**
     * @notice Get the linked identity registry address.
     */
    function getIdentityRegistry() external view returns (address) {
        return identityRegistry;
    }
}
