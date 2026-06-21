-- Agent decisions table for multi-agent v2 bots
create table if not exists agent_decisions (
  id uuid default gen_random_uuid() primary key,
  timestamp timestamptz default now(),
  bot_id text not null,
  ticker text not null,
  sentiment_score numeric,
  sentiment_label text,
  technical_score numeric,
  setup_kwaliteit text,
  risico_score numeric,
  risico_label text,
  finale_beslissing text,
  consensus text,
  confidence numeric,
  doorslaggevende_factor text,
  reden text,
  trade_uitgevoerd boolean default false,
  agents_failed text[],
  llm_used boolean default true,
  created_at timestamptz default now()
);

-- Index for common queries
create index if not exists idx_agent_decisions_bot_ticker
  on agent_decisions(bot_id, ticker, timestamp desc);

-- Enable RLS (disable for service role access)
alter table agent_decisions enable row level security;
