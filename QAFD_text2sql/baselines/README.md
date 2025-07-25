# Text-to-SQL Baselines

This directory contains implementations of four text-to-SQL baselines:

## Available Baselines

### 1. CodeS
Uses CodeS-7B model with schema item classifier for enhanced understanding.
- **Directory**: `codes/` - See [codes/README.md](codes/README.md) for setup

### 2. DailSQL  
Utilizes GPT-4o with Code Representation (CR) prompting approach.
- **Directory**: `dailsql/` - See [dailsql/README.md](dailsql/README.md) for setup

### 3. CHESS
Multi-step reasoning framework with information retrieval and candidate generation.
- **Directory**: `CHESS_spider2/` - See [CHESS_spider2/README.md](CHESS_spider2/README.md) for setup

### 4. DIN-SQL
Python-based implementation with OpenAI API integration.
- **Directory**: `dinsql/` - See [dinsql/README.md](dinsql/README.md) for setup

## Directory Structure

```
baselines/
├── codes/           # CodeS implementation
├── dailsql/         # DailSQL implementation  
├── CHESS_spider2/   # CHESS implementation
├── dinsql/          # DIN-SQL implementation
└── README.md        # This file
```

For installation and usage instructions, refer to the README file in each baseline's directory.
