# Multi-Agent Negotiation Simulator

A domain-agnostic, generative AI-powered multi-agent negotiation framework built for Bytecamp '26.

## Overview
Autonomous agents (buyers, sellers, mediators) negotiate over shared resources in a decentralised environment — without a central controller. Agent decisions are powered by OpenAI GPT-4o.

## Supported Domains
| Domain | Resource | Participants |
|---|---|---|
| `logistics` | Truck capacity (pallets) | Supplier-A, Supplier-B, Transporter, Mediator |
| `cloud` | Compute units (vCPUs) | ServiceA, ServiceB, CloudProvider, Mediator |
| `finance` | Asset price (USD) | BuyerFund, SellerFirm, Mediator |

## Project Structure
```
/core
  agent.py          ← Generic Agent class + LLM evaluate()
  negotiation.py    ← NegotiationSession + round loop
  environment.py    ← Domain config loader + session initializer

/domains
  logistics.json    ← Logistics scenario config
  cloud.json        ← Cloud resource scenario config
  finance.json      ← Financial market scenario config

/backend
  main.py           ← FastAPI app (REST + WebSocket)

/frontend
  ...               ← React + TypeScript + Vite + TailwindCSS
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run backend
```bash
cd backend
uvicorn main:app --reload
```

## Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

## Tech Stack
- **Backend**: Python, FastAPI, asyncio
- **AI**: OpenAI GPT-4o
- **Frontend**: React + TypeScript + Vite + TailwindCSS + Recharts
- **Communication**: REST + WebSocket
