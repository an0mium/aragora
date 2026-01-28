/**
 * Decision Receipts Example
 *
 * Demonstrates how to work with decision receipts using the Aragora SDK.
 * Includes listing, verification, statistics, and export operations.
 *
 * Usage:
 *   npx ts-node examples/receipts-demo.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import { createClient } from '@aragora/sdk';
import * as fs from 'fs';

async function main() {
  // Create client
  const client = createClient({
    baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('=== Decision Receipts Demo ===\n');

  // ---------------------------------------------------------------------------
  // 1. List receipts
  // ---------------------------------------------------------------------------
  console.log('1. Listing recent decision receipts...');
  const { receipts } = await client.receipts.list({ limit: 5 });
  console.log(`   Found ${receipts.length} receipts`);

  if (receipts.length === 0) {
    console.log('   No receipts found. Run a debate first to generate receipts.');
    return;
  }

  for (const receipt of receipts) {
    console.log(`   - ${receipt.receipt_id}: ${receipt.verdict} (confidence: ${receipt.confidence})`);
  }

  // ---------------------------------------------------------------------------
  // 2. Get receipt details
  // ---------------------------------------------------------------------------
  console.log('\n2. Getting receipt details...');
  const receiptId = receipts[0].receipt_id;
  const receipt = await client.receipts.get(receiptId);
  console.log(`   ID: ${receipt.receipt_id}`);
  console.log(`   Verdict: ${receipt.verdict}`);
  console.log(`   Confidence: ${receipt.confidence}`);
  console.log(`   Consensus: ${receipt.consensus_reached ? 'Reached' : 'Not reached'}`);
  console.log(`   Participating Agents: ${receipt.participating_agents?.join(', ')}`);

  // Check for dissent
  if (client.receipts.hasDissent(receipt)) {
    console.log(`   Dissenting Agents: ${receipt.dissenting_agents?.join(', ')}`);
  }

  // ---------------------------------------------------------------------------
  // 3. Verify receipt integrity
  // ---------------------------------------------------------------------------
  console.log('\n3. Verifying receipt integrity...');
  const verification = await client.receipts.verify(receiptId);
  console.log(`   Valid: ${verification.valid}`);
  console.log(`   Hash: ${verification.hash}`);
  console.log(`   Verified at: ${verification.verified_at}`);

  // ---------------------------------------------------------------------------
  // 4. Batch verification (if multiple receipts)
  // ---------------------------------------------------------------------------
  if (receipts.length > 1) {
    console.log('\n4. Batch verification of multiple receipts...');
    const receiptIds = receipts.slice(0, 3).map((r) => r.receipt_id);
    const batchResult = await client.receipts.verifyBatch(receiptIds);
    console.log(`   Verified: ${batchResult.total_verified}`);
    console.log(`   Valid: ${batchResult.total_valid}`);
    console.log(`   Invalid: ${batchResult.total_invalid}`);

    for (const result of batchResult.results) {
      const status = result.valid ? 'VALID' : `INVALID: ${result.error}`;
      console.log(`   - ${result.receipt_id}: ${status}`);
    }
  }

  // ---------------------------------------------------------------------------
  // 5. Get receipt statistics
  // ---------------------------------------------------------------------------
  console.log('\n5. Receipt statistics...');
  const stats = await client.receipts.getStats();
  console.log(`   Total receipts: ${stats.total_count}`);
  console.log(`   Consensus rate: ${(stats.consensus_rate * 100).toFixed(1)}%`);
  console.log(`   Average confidence: ${(stats.average_confidence * 100).toFixed(1)}%`);
  console.log('   By verdict:');
  for (const [verdict, count] of Object.entries(stats.by_verdict)) {
    console.log(`     - ${verdict}: ${count}`);
  }

  // ---------------------------------------------------------------------------
  // 6. Export receipt as PDF
  // ---------------------------------------------------------------------------
  console.log('\n6. Exporting receipt as PDF...');
  try {
    const pdfBlob = await client.receipts.exportPdf(receiptId);
    const pdfBuffer = Buffer.from(await pdfBlob.arrayBuffer());
    const pdfPath = `./receipt-${receiptId}.pdf`;
    fs.writeFileSync(pdfPath, pdfBuffer);
    console.log(`   Saved to: ${pdfPath}`);
  } catch (error) {
    console.log(`   PDF export not available: ${error}`);
  }

  // ---------------------------------------------------------------------------
  // 7. Export multiple receipts as CSV
  // ---------------------------------------------------------------------------
  console.log('\n7. Exporting receipts as CSV...');
  try {
    const receiptIds = receipts.map((r) => r.receipt_id);
    const csvBlob = await client.receipts.exportCsv(receiptIds);
    const csvText = await csvBlob.text();
    const csvPath = './receipts-export.csv';
    fs.writeFileSync(csvPath, csvText);
    console.log(`   Saved to: ${csvPath}`);
    console.log('   Preview:');
    const lines = csvText.split('\n').slice(0, 3);
    for (const line of lines) {
      console.log(`     ${line}`);
    }
  } catch (error) {
    console.log(`   CSV export not available: ${error}`);
  }

  // ---------------------------------------------------------------------------
  // 8. Gauntlet receipts (if available)
  // ---------------------------------------------------------------------------
  console.log('\n8. Gauntlet receipts...');
  try {
    const { receipts: gauntletReceipts } = await client.receipts.listGauntlet({ limit: 3 });
    console.log(`   Found ${gauntletReceipts.length} gauntlet receipts`);

    if (gauntletReceipts.length > 0) {
      const gauntletId = gauntletReceipts[0].receipt_id;

      // Export as different formats
      for (const format of ['markdown', 'html', 'sarif'] as const) {
        const exported = await client.receipts.exportGauntlet(gauntletId, format);
        console.log(`   Exported as ${format}: ${exported.content.length} bytes`);
      }
    }
  } catch (error) {
    console.log(`   Gauntlet receipts not available: ${error}`);
  }

  console.log('\n=== Demo Complete ===');
}

main().catch((error) => {
  console.error('Error:', error.message);
  process.exit(1);
});
