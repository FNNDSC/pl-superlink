# FedMed Flower SuperLink

This repo contains the complete _ChRIS_ plugin that launches the Flower SuperLink for the FedMed demo. Run the commands below from inside `plugins/superlink_plugin/`. The SuperLink coordinates training, waits for a configurable number of Flower SuperNodes to connect over gRPC, and writes a JSON summary of the run and the weights file to `/outgoing`.

## Build
```bash
docker build -t fedmed/pl-superlink .
```

## Run (example)
```bash
docker run --rm --name fedmed-superlink --network fedmed-net \
  -v $(pwd)/demo/server-in:/incoming:ro \
  -v $(pwd)/demo/server-out:/outgoing:rw \
  fedmed/pl-superlink \
    fedmed-pl-superlink \
      --rounds 3 \
      --total-clients 3 \
      --local-epochs 10 \
      --learning-rate 0.2 \
      --data-seed 13 \
      --fraction-evaluate 1.0 \
      /incoming /outgoing
```

Use `docker inspect fedmed-superlink` to obtain the IPv4 address and pass it to SuperNode containers (Flower prefers literal addresses on Docker networks).

### Accepted parameters
- `--rounds` (default: 3)
- `--total-clients` (default: 3)
- `--local-epochs` (default: 10)
- `--learning-rate` (default: 0.2)
- `--data-seed` (default: 13)
- `--fraction-evaluate` (default: 1.0)

All other networking, federation, and state options use baked-in defaults (host `0.0.0.0`, Fleet `9092`, Control `9093`, ServerApp `9091`, summary `server_summary.json`) and are no longer exposed as CLI flags.
