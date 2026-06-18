-- ============================================================
-- Supabase setup voor Hybrid Trading Bot
-- Voer dit uit via Supabase Dashboard → SQL Editor
-- ============================================================

-- ── Portfolio state tabel ─────────────────────────────────────
-- Elke run schrijft één nieuwe rij. De bot leest de meest recente.
create table if not exists portfolio_state (
  id               uuid        default gen_random_uuid() primary key,
  timestamp        timestamptz default now(),
  capital_eur      numeric     not null,
  open_positions   jsonb       not null default '{}'::jsonb,
  total_trades     int         not null default 0,
  wins             int         not null default 0,
  gross_win        numeric     not null default 0,
  gross_loss       numeric     not null default 0,
  win_rate         numeric     not null default 0,
  profit_factor    numeric     not null default 0,
  total_return_pct numeric     not null default 0
);

-- Index voor snelle ophaling van meest recente state
create index if not exists idx_portfolio_state_timestamp
  on portfolio_state (timestamp desc);

-- ── Trades tabel ──────────────────────────────────────────────
-- Elke BUY en SELL actie wordt hier gelogd.
-- Exit-velden (exit_price, pnl_*) zijn null bij BUY rows.
create table if not exists trades (
  id                    uuid        default gen_random_uuid() primary key,
  timestamp             timestamptz default now(),
  ticker                text        not null,
  action                text        not null check (action in ('BUY','SELL')),
  price                 numeric     not null,
  position_size_pct     numeric,
  position_size_eur     numeric,
  entry_price           numeric,
  exit_price            numeric,
  pnl_pct               numeric,
  pnl_eur               numeric,
  exit_reason           text,
  regime                text,
  strategy              text        check (strategy in ('v4','mr','sdbf')),
  llm_beslissing        text,
  llm_confidence        numeric,
  llm_nieuws_sentiment  text,
  llm_reden             text
);

create index if not exists idx_trades_timestamp on trades (timestamp desc);
create index if not exists idx_trades_ticker    on trades (ticker);

-- ── Signals tabel ─────────────────────────────────────────────
-- Elke technische entry kans wordt gelogd, ook als LLM hem blokkeert.
create table if not exists signals (
  id              uuid        default gen_random_uuid() primary key,
  timestamp       timestamptz default now(),
  ticker          text        not null,
  regime          text,
  rsi             numeric,
  macd_hist       numeric,
  atr_ratio       numeric,
  volume_ratio    numeric,
  ma200_slope     numeric,
  trailing_stop   numeric,
  signal_type     text        check (signal_type in ('v4','mr','sdbf')),
  llm_beslissing  text,
  llm_confidence  numeric,
  entry_blocked   boolean     default false,
  block_reason    text
);

create index if not exists idx_signals_timestamp on signals (timestamp desc);
create index if not exists idx_signals_ticker    on signals (ticker);

-- ── Row Level Security ─────────────────────────────────────────
-- Bot gebruikt service-role key (SUPABASE_KEY), die RLS omzeilt.
-- Schakel RLS in voor veiligheid bij publieke projecten.
alter table portfolio_state enable row level security;
alter table trades          enable row level security;
alter table signals         enable row level security;

-- Service role heeft volledige toegang (standaard in Supabase).
-- Geen extra policies nodig als je alleen de service key gebruikt.

-- ── Handige views ─────────────────────────────────────────────

-- Laatste portfolio state
create or replace view v_portfolio_latest as
  select * from portfolio_state
  order by timestamp desc
  limit 1;

-- Alle closed trades (SELL rows met pnl)
create or replace view v_closed_trades as
  select
    timestamp,
    ticker,
    strategy,
    regime,
    entry_price,
    exit_price,
    pnl_pct,
    pnl_eur,
    exit_reason,
    llm_beslissing,
    llm_nieuws_sentiment
  from trades
  where action = 'SELL'
  order by timestamp desc;

-- Performance per ticker
create or replace view v_performance_per_ticker as
  select
    ticker,
    count(*)                                       as n_trades,
    round(avg(case when pnl_eur > 0 then 1.0 else 0.0 end) * 100, 1)
                                                   as win_rate_pct,
    round(sum(pnl_eur), 2)                         as total_pnl_eur,
    round(
      sum(case when pnl_eur > 0 then pnl_eur else 0 end) /
      nullif(sum(case when pnl_eur < 0 then abs(pnl_eur) else 0 end), 0),
      2)                                           as profit_factor
  from trades
  where action = 'SELL'
  group by ticker
  order by total_pnl_eur desc;

-- LLM effectiviteit: terecht vs onterecht geblokkeerd
-- (alleen zichtbaar als je ook de counterfactual zou bijhouden;
--  hier tonen we geblokkeerde signalen)
create or replace view v_llm_blocks as
  select
    timestamp,
    ticker,
    regime,
    signal_type,
    llm_confidence,
    block_reason
  from signals
  where entry_blocked = true
  order by timestamp desc;
