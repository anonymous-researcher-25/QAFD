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

## Remark
DailSQL, DIN-SQL, and CodeS require adaptations of the Spider 2 Snowflake dataset to ensure compatibility with these methods. Due to GitHub's file size limitations, the modified dataset is hosted externally and can be accessed via
[https://drive.google.com/file/d/1Y1zSbZ9cBUVJFI_W3t5EZZTzcUi1Ma4i/view](resource). For more information on the original version of the Spider 2 dataset, please refer to the official repository:
[https://github.com/xlang-ai/Spider2](Spider2)
