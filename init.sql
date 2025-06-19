-- MarketPulse Database Initialization Script

-- Create database if not exists (handled by Docker environment)

-- Create tables for MarketPulse
CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS economic_indicators (
    id SERIAL PRIMARY KEY,
    indicator_code VARCHAR(50) NOT NULL,
    indicator_name VARCHAR(200),
    date DATE NOT NULL,
    value DECIMAL(15,4),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(indicator_code, date)
);

CREATE TABLE IF NOT EXISTS crypto_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    price DECIMAL(15,8),
    market_cap BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50),
    asset_symbol VARCHAR(10),
    prediction_value DECIMAL(10,4),
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stock_symbol_date ON stock_prices(symbol, date);
CREATE INDEX IF NOT EXISTS idx_stock_created_at ON stock_prices(created_at);

CREATE INDEX IF NOT EXISTS idx_econ_code_date ON economic_indicators(indicator_code, date);
CREATE INDEX IF NOT EXISTS idx_econ_created_at ON economic_indicators(created_at);

CREATE INDEX IF NOT EXISTS idx_crypto_symbol_date ON crypto_prices(symbol, date);
CREATE INDEX IF NOT EXISTS idx_crypto_created_at ON crypto_prices(created_at);

CREATE INDEX IF NOT EXISTS idx_pred_model_symbol ON predictions(model_type, asset_symbol);
CREATE INDEX IF NOT EXISTS idx_pred_created_at ON predictions(created_at);

-- Insert some sample data for testing
INSERT INTO stock_prices (symbol, date, open_price, high_price, low_price, close_price, volume) 
VALUES 
    ('AAPL', '2025-01-01', 150.00, 155.00, 148.00, 152.50, 1000000),
    ('GOOGL', '2025-01-01', 2800.00, 2850.00, 2780.00, 2820.00, 500000)
ON CONFLICT (symbol, date) DO NOTHING;

INSERT INTO economic_indicators (indicator_code, indicator_name, date, value)
VALUES
    ('GDP', 'Gross Domestic Product', '2024-12-01', 25000.0),
    ('UNRATE', 'Unemployment Rate', '2024-12-01', 3.8)
ON CONFLICT (indicator_code, date) DO NOTHING;

INSERT INTO crypto_prices (symbol, date, price, market_cap)
VALUES
    ('BTC', '2025-01-01', 45000.00, 850000000000),
    ('ETH', '2025-01-01', 3200.00, 380000000000)
ON CONFLICT (symbol, date) DO NOTHING;
