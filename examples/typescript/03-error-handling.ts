/**
 * Example 03: Error Handling
 *
 * This example demonstrates best practices for handling errors
 * when using the Aragora TypeScript SDK.
 */

import { AragoraClient, AragoraError } from "@aragora/sdk";

// Initialize the client with retry configuration
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL || "http://localhost:8080",
  apiKey: process.env.ARAGORA_API_KEY,
  timeout: 30000,
  retry: {
    maxRetries: 3,
    initialDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
    jitter: true,
  },
});

// Example 1: Basic error handling
async function basicErrorHandling() {
  console.log("=== Example 1: Basic Error Handling ===\n");

  try {
    // Try to get a non-existent debate
    await client.debates.get("non-existent-debate-id");
  } catch (error) {
    if (error instanceof AragoraError) {
      console.log("AragoraError caught:");
      console.log("  Code:", error.code);
      console.log("  Status:", error.status);
      console.log("  Message:", error.message);
      console.log("  Retryable:", error.retryable);
      console.log("  User Message:", error.toUserMessage());
    } else {
      // Handle unexpected errors
      console.error("Unexpected error:", error);
    }
  }
  console.log("");
}

// Example 2: Handling specific error types
async function handleSpecificErrors() {
  console.log("=== Example 2: Handling Specific Error Types ===\n");

  try {
    await client.debates.create({
      task: "Test debate",
      agents: ["invalid-agent"],
    });
  } catch (error) {
    if (error instanceof AragoraError) {
      switch (error.code) {
        case "UNAUTHORIZED":
          console.log("Authentication required. Please provide a valid API key.");
          break;
        case "FORBIDDEN":
          console.log("Permission denied. Check your role and permissions.");
          break;
        case "NOT_FOUND":
          console.log("Resource not found. It may have been deleted.");
          break;
        case "VALIDATION_ERROR":
          console.log("Invalid request data:", error.message);
          break;
        case "RATE_LIMITED":
          console.log("Rate limited. Wait before retrying.");
          break;
        case "TIMEOUT":
          console.log("Request timed out. The server may be overloaded.");
          break;
        case "NETWORK_ERROR":
          console.log("Network error. Check your connection.");
          break;
        default:
          console.log("API error:", error.toUserMessage());
      }
    } else {
      throw error;
    }
  }
  console.log("");
}

// Example 3: Retry with custom logic
async function customRetryLogic() {
  console.log("=== Example 3: Custom Retry Logic ===\n");

  const maxAttempts = 3;
  let attempt = 0;

  while (attempt < maxAttempts) {
    try {
      attempt++;
      console.log(`Attempt ${attempt}/${maxAttempts}...`);

      // Disable built-in retry for custom retry logic
      const result = await client.debates.run(
        {
          task: "Quick test debate",
          agents: ["anthropic-api"],
          rounds: 1,
        },
        { retry: false }
      );

      console.log("Success! Debate ID:", result.debate_id);
      break;
    } catch (error) {
      if (error instanceof AragoraError && error.retryable) {
        if (attempt < maxAttempts) {
          const delay = Math.pow(2, attempt) * 1000;
          console.log(`Retryable error, waiting ${delay}ms before retry...`);
          await new Promise((resolve) => setTimeout(resolve, delay));
        } else {
          console.log("Max retries reached");
          throw error;
        }
      } else {
        // Non-retryable error, don't retry
        console.log("Non-retryable error:", (error as Error).message);
        throw error;
      }
    }
  }
  console.log("");
}

// Example 4: Graceful degradation
async function gracefulDegradation() {
  console.log("=== Example 4: Graceful Degradation ===\n");

  interface DebateResult {
    source: "api" | "cache" | "fallback";
    data: unknown;
  }

  // Simulated cache
  const cache = new Map<string, unknown>();

  async function getDebateWithFallback(debateId: string): Promise<DebateResult> {
    // Try API first
    try {
      const debate = await client.debates.get(debateId);
      // Update cache on success
      cache.set(debateId, debate);
      return { source: "api", data: debate };
    } catch (error) {
      if (error instanceof AragoraError) {
        // Check cache for transient errors
        if (error.retryable && cache.has(debateId)) {
          console.log("Using cached data due to transient error");
          return { source: "cache", data: cache.get(debateId) };
        }

        // Return fallback data for 404
        if (error.code === "NOT_FOUND") {
          return {
            source: "fallback",
            data: {
              debate_id: debateId,
              status: "not_found",
              message: "Debate not found or has been deleted",
            },
          };
        }
      }
      throw error;
    }
  }

  // Test the fallback mechanism
  try {
    const result = await getDebateWithFallback("test-debate-123");
    console.log("Result source:", result.source);
    console.log("Result data:", result.data);
  } catch (error) {
    console.log("Failed to get debate:", (error as Error).message);
  }
  console.log("");
}

// Example 5: Timeout handling
async function timeoutHandling() {
  console.log("=== Example 5: Timeout Handling ===\n");

  try {
    // Create a client with very short timeout for demonstration
    const shortTimeoutClient = new AragoraClient({
      baseUrl: process.env.ARAGORA_API_URL || "http://localhost:8080",
      apiKey: process.env.ARAGORA_API_KEY,
      timeout: 100, // Very short timeout
      retry: { maxRetries: 0 },
    });

    await shortTimeoutClient.debates.list({ limit: 10 });
  } catch (error) {
    if (error instanceof AragoraError && error.code === "TIMEOUT") {
      console.log("Request timed out");
      console.log("Consider:");
      console.log("  - Increasing timeout for this operation");
      console.log("  - Using pagination with smaller page sizes");
      console.log("  - Checking server health");
    }
  }
  console.log("");
}

// Example 6: Validation error handling
async function validationErrorHandling() {
  console.log("=== Example 6: Validation Error Handling ===\n");

  try {
    // Intentionally send invalid data
    await client.debates.create({
      task: "", // Empty task
      agents: [], // Empty agents array
      rounds: -1, // Invalid rounds
    });
  } catch (error) {
    if (error instanceof AragoraError && error.code === "VALIDATION_ERROR") {
      console.log("Validation failed:");
      console.log("  Message:", error.message);

      // In a real app, you might parse validation details
      // and show field-specific errors to the user
    }
  }
  console.log("");
}

// Run all examples
async function main() {
  console.log("Error Handling Examples\n");
  console.log("=======================\n");

  await basicErrorHandling();
  await handleSpecificErrors();
  await gracefulDegradation();
  await timeoutHandling();
  await validationErrorHandling();

  // Only run custom retry logic if API is available
  try {
    await client.health();
    await customRetryLogic();
  } catch {
    console.log("Skipping custom retry example (API not available)\n");
  }

  console.log("Error handling examples completed!");
}

main().catch(console.error);
