use std::env::args;
use std::error::Error;

use bitcoincore_rpc::Auth;
use nockchain_bitcoin_sync::{BitcoinRPCConnection, BitcoinWatcher};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let block: u64 = args().collect::<Vec<_>>()[1].parse()?;
    let connection = BitcoinRPCConnection::new(
        "http://127.0.0.1:8332".to_string(),
        /*
        Auth::CookieFile(PathBuf::from(
            "cookiefile",
        )),*/
        Auth::None,
        block,
    );
    let watcher = BitcoinWatcher::new(connection).await?;
    let block_ref = watcher.watch().await?;
    println!("Block {} at height {}", block_ref.hash, block_ref.height);
    Ok(())
}
