# Nockchain

**Nockchain is a lightweight blockchain for heavyweight verifiable applications.**


We believe the future of blockchains is lightweight trustless settlement of heavyweight verifiable computation. The only way to get there is by replacing verifiability-via-public-replication with verifiability-via-private-proving. Proving happens off-chain; verification is on-chain.

*Nockchain is entirely experimental and many parts are unaudited. We make no representations or guarantees as to the behavior of this software.*


## Setup

Install `rustup` by following their instructions at: [https://rustup.rs/](https://rustup.rs/)

Install `hoonc`, the Hoon compiler:

```
make install-hoonc
```

To build the Nockchain and the wallet binaries and their required assets:

```
make build
```

## Install Wallet

After you've run the setup and build commands, install the wallet:

```
make install-nockchain-wallet
```

To run the wallet, see the nockchain-wallet [README](./crates/nockchain-wallet/README.md).


## Install Nockchain

After you've run the setup and build commands, install the wallet:

```
make install-nockchain
```


## Testing Nodes

To run a test Nockchain node that publishes the genesis block:

```
make run-nockchain-leader
```


To run a test Nockchain node that waits for the genesis block:

```
make run-nockchain-follower
```

