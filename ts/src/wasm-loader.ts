/**
 * WASM module loader for rustful-ts
 *
 * Compatible with Node.js, Bun, and browsers.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let wasmModule: any = null;
let wasmInitialized = false;
let initPromise: Promise<void> | null = null;

/**
 * Detect runtime environment
 */
type Runtime = 'bun' | 'node' | 'browser';

function detectRuntime(): Runtime {
  // @ts-expect-error Bun global
  if (typeof Bun !== 'undefined') {
    return 'bun';
  }
  if (typeof process !== 'undefined' && process.versions?.node) {
    return 'node';
  }
  return 'browser';
}

/**
 * Initialize the WASM module
 *
 * This must be called before using any of the prediction algorithms.
 * It's safe to call multiple times - subsequent calls will return immediately.
 *
 * @example
 * ```typescript
 * import { initWasm, Arima } from 'rustful-ts';
 *
 * await initWasm();
 * const model = new Arima(1, 1, 1);
 * ```
 */
export async function initWasm(): Promise<void> {
  if (wasmInitialized) {
    return;
  }

  if (initPromise) {
    return initPromise;
  }

  initPromise = (async () => {
    const runtime = detectRuntime();

    try {
      // Try to import the WASM module
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const wasm = await import('../pkg/rustful_wasm.js').catch(() =>
        // Fallback to old name for backwards compatibility
        import('../pkg/rustful_ts.js')
      );

      // Initialize the WASM module
      if (typeof wasm.default === 'function') {
        await wasm.default();
      }
      wasmModule = wasm;
      wasmInitialized = true;
    } catch (error) {
      initPromise = null;
      const runtime_name = runtime.charAt(0).toUpperCase() + runtime.slice(1);
      throw new Error(
        `Failed to initialize WASM in ${runtime_name}: ${error}\n` +
        `Make sure you've built the WASM module: npm run build:wasm`
      );
    }
  })();

  return initPromise;
}

/**
 * Initialize with custom WASM path (for bundlers or custom setups)
 *
 * @param wasmPath - Path or URL to the .wasm file
 */
export async function initWasmFromPath(wasmPath: string): Promise<void> {
  if (wasmInitialized) {
    return;
  }

  try {
    const wasm = await import('../pkg/rustful_wasm.js').catch(() =>
      import('../pkg/rustful_ts.js')
    );

    // For custom paths, fetch the wasm binary
    if (typeof fetch !== 'undefined') {
      const response = await fetch(wasmPath);
      const buffer = await response.arrayBuffer();
      await wasm.default(buffer);
    } else {
      // Node.js/Bun: read file
      const fs = await import('fs/promises');
      const buffer = await fs.readFile(wasmPath);
      await wasm.default(buffer);
    }

    wasmModule = wasm;
    wasmInitialized = true;
  } catch (error) {
    throw new Error(`Failed to load WASM from ${wasmPath}: ${error}`);
  }
}

/**
 * Check if the WASM module is ready
 */
export function isWasmReady(): boolean {
  return wasmInitialized;
}

/**
 * Get current runtime
 */
export function getRuntime(): Runtime {
  return detectRuntime();
}

/**
 * Get the WASM module (internal use)
 * @internal
 */
export function getWasmModule(): unknown {
  if (!wasmInitialized) {
    throw new Error(
      'WASM module not initialized. Call initWasm() first.'
    );
  }
  return wasmModule;
}

/**
 * Ensure WASM is initialized before using algorithms
 * @internal
 */
export async function ensureWasm(): Promise<void> {
  if (!wasmInitialized) {
    await initWasm();
  }
}

/**
 * Reset WASM state (for testing)
 * @internal
 */
export function resetWasm(): void {
  wasmModule = null;
  wasmInitialized = false;
  initPromise = null;
}
