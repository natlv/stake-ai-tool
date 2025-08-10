// DuckDB-WASM singleton with CDN bundles
import duckdb from "@duckdb/duckdb-wasm";

let dbPromise: Promise<duckdb.AsyncDuckDB> | null = null;

export async function getDuckDB(): Promise<duckdb.AsyncDuckDB> {
  if (dbPromise) return dbPromise;
  dbPromise = (async () => {
    const bundles = duckdb.getJsDelivrBundles();
    const bundle = await duckdb.selectBundle(bundles);
    const worker = new Worker(bundle.mainWorker);
    const logger = new duckdb.ConsoleLogger();
    const db = new duckdb.AsyncDuckDB(logger, worker);
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    return db;
  })();
  return dbPromise;
}

export async function withConnection<T>(fn: (conn: duckdb.AsyncDuckDBConnection) => Promise<T>): Promise<T> {
  const db = await getDuckDB();
  const conn = await db.connect();
  try {
    return await fn(conn);
  } finally {
    await conn.close();
  }
}
