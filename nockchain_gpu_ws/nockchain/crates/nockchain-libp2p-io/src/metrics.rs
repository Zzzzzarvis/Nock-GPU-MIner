use gnort::*;

metrics_struct![
    NockchainP2PMetrics,
    (gossip_acked, "nockchain-libp2p-io.gossip_acked", Count),
    (gossip_nacked, "nockchain-libp2p-io.gossip_nacked", Count),
    (gossip_erred, "nockchain-libp2p-io.gossip_erred", Count),
    (gossip_dropped, "nockchain-libp2p-io.gossip_dropped", Count),
    (requests_peeked_some, "nockchain-libp2p-io.requests_peeked_some", Count),
    (requests_peeked_none, "nockchain-libp2p-io.requests_peeked_none", Count),
    (requests_erred, "nockchain-libp2p-io.requests_erred", Count),
    (requests_dropped, "nockchain-libp2p-io.requests_dropped", Count),
    (responses_acked, "nockchain-libp2p-io.responses_acked", Count),
    (responses_nacked, "nockchain-libp2p-io.responses_nacked", Count),
    (responses_erred, "nockchain-libp2p-io.responses_erred", Count),
    (responses_dropped, "nockchain-libp2p-io.responses_dropped", Count),
    (block_request_cache_hits, "nockchain-libp2p-io.block_request_cache_hits", Count),
    (tx_request_cache_hits, "nockchain-libp2p-io.tx_request_cache_hits", Count),
    (block_seen_cache_hits, "nockchain-libp2p-io.block_seen_cache_hits", Count),
    (tx_seen_cache_hits, "nockchain-libp2p-io.tx_seen_cache_hits", Count),
    (block_request_cache_misses, "nockchain-libp2p-io.block_request_cache_misses", Count),
    (tx_request_cache_misses, "nockchain-libp2p-io.tx_request_cache_misses", Count),
    (block_seen_cache_misses, "nockchain-libp2p-io.block_seen_cache_misses", Count),
    (tx_seen_cache_misses, "nockchain-libp2p-io.tx_seen_cache_misses", Count)
];
