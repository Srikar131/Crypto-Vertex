'use client';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import axios from 'axios';

type PredictionData = {
  predictions: {
    lstm: number;
    bidirectional_lstm: number;
    xgboost: number;
    ensemble: number;
  };
};
type HistoryData = { Date: string; Close: number };

const TICKERS = [
  { symbol: 'btc', name: 'BTC', color: 'text-yellow-600 dark:text-yellow-400' },
  { symbol: 'eth', name: 'ETH', color: 'text-blue-600 dark:text-blue-400' },
  { symbol: 'sol', name: 'SOL', color: 'text-teal-600 dark:text-teal-400' },
  { symbol: 'ada', name: 'ADA', color: 'text-indigo-600 dark:text-indigo-400' },
  { symbol: 'doge', name: 'DOGE', color: 'text-orange-500 dark:text-orange-300' }
];

const predictionModels = [
  { key: 'ensemble', label: 'Ensemble AI', accent: 'border-purple-300 text-purple-700 dark:border-purple-500 dark:text-purple-200', emoji: '🤖' },
  { key: 'lstm', label: 'LSTM', accent: 'border-blue-200 text-blue-700 dark:border-blue-500 dark:text-blue-200', emoji: '🧠' },
  { key: 'bidirectional_lstm', label: 'Bi-LSTM', accent: 'border-teal-200 text-teal-700 dark:border-teal-500 dark:text-teal-200', emoji: '🔁' },
  { key: 'xgboost', label: 'XGBoost', accent: 'border-orange-200 text-orange-700 dark:border-orange-300 dark:text-orange-300', emoji: '⚡' }
];

const formatPrice = (price: number | undefined) =>
  price === undefined ? '$...' : '$' + price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const fetchLatestNews = async (coin: string) => {
  const symbol = coin.toLowerCase().split('-')[0];
  let newsData: any[] = [];
  try {
    const res = await fetch(`/api/cryptonews?symbol=${symbol}`);
    const data = await res.json();
    newsData = data.results?.slice(0, 5) || [];
  } catch {
    newsData = [];
  }
  return newsData;
};

export default function PredictionDashboard() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoryData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCrypto, setSelectedCrypto] = useState('BTC-USD');
  const [news, setNews] = useState<any[]>([]);
  const [newsLoading, setNewsLoading] = useState(false);
  const [history, setHistory] = useState<{ coin: string; price: string; verdict: string; }[]>([]);
  const [modelModal, setModelModal] = useState<{ open: boolean; key: string }>({ open: false, key: '' });
  const [modelHistory, setModelHistory] = useState<{ [key: string]: number[] }>({
    ensemble: [], lstm: [], bidirectional_lstm: [], xgboost: []
  });

  // THEME TOGGLE
  const [dark, setDark] = useState(false);

  // Live Ticker State (direct CoinGecko, harmless console error)
  const [tickerData, setTickerData] = useState<{ [k: string]: { price: number; prev: number } }>({
    btc: { price: 0, prev: 0 },
    eth: { price: 0, prev: 0 },
    sol: { price: 0, prev: 0 },
    ada: { price: 0, prev: 0 },
    doge: { price: 0, prev: 0 },
  });
  const marqueeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // NOTE: If you want zero CORS errors in browser, comment out this fetch and fetch prices via backend
    async function getTicker() {
      try {
        const res = await fetch(
          'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana,cardano,dogecoin&vs_currencies=usd'
        );
        const json = await res.json();
        setTickerData(prev => ({
          btc: { price: json.bitcoin.usd, prev: prev.btc.price || json.bitcoin.usd },
          eth: { price: json.ethereum.usd, prev: prev.eth.price || json.ethereum.usd },
          sol: { price: json.solana.usd, prev: prev.sol.price || json.solana.usd },
          ada: { price: json.cardano.usd, prev: prev.ada.price || json.cardano.usd },
          doge: { price: json.dogecoin.usd, prev: prev.doge.price || json.dogecoin.usd }
        }));
      } catch { }
    }
    getTicker();
    const int = setInterval(getTicker, 30000);
    return () => clearInterval(int);
  }, []);

  const cryptoOptions = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD'];
  // FIXED: fallback port now 8000, not 10000
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setHistoricalData([]);
    try {
      const requestBody = { symbol: selectedCrypto };
      const [predResponse, histResponse] = await Promise.all([
        axios.post(`${API_BASE_URL}/predict`, requestBody),
        axios.post(`${API_BASE_URL}/history`, requestBody)
      ]);
      setPrediction(predResponse.data);
      setHistoricalData(histResponse.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "An unexpected error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [selectedCrypto, API_BASE_URL]);

  useEffect(() => {
    fetchData();

    async function getNews() {
      setNewsLoading(true);
      const newsData = await fetchLatestNews(selectedCrypto);
      setNews(newsData);
      setNewsLoading(false);
    }
    getNews();
  }, [selectedCrypto, fetchData]);

  // Marquee Animation
  const tickerContent = (
    <div className="flex items-center gap-10 text-[1rem]">
      {TICKERS.map(({ symbol, name, color }) => (
        <div key={symbol} className="flex items-center gap-1 min-w-[77px]">
          <span className={`font-bold ${color}`}>{name}</span>
          <span className={
            tickerData[symbol].price > tickerData[symbol].prev
              ? 'text-green-600 dark:text-green-400 font-semibold'
              : tickerData[symbol].price < tickerData[symbol].prev
                ? 'text-red-500 dark:text-red-400 font-semibold'
                : 'text-gray-800 dark:text-gray-200 font-semibold'
          }>
            {tickerData[symbol].price ? `$${tickerData[symbol].price.toLocaleString('en-US', { minimumFractionDigits: 2 })}` : '...'}
          </span>
          {tickerData[symbol].price > tickerData[symbol].prev && <span className="text-green-500 dark:text-green-400">▲</span>}
          {tickerData[symbol].price < tickerData[symbol].prev && <span className="text-red-500 dark:text-red-400">▼</span>}
        </div>
      ))}
    </div>
  );

  // Verdict/Confidence logic & history/model history update
  useEffect(() => {
    if (!prediction) return;
    const preds = prediction.predictions;
    const avg = (preds.lstm + preds.xgboost + preds.bidirectional_lstm) / 3;
    let verdict = 'Neutral';
    if (preds.ensemble > avg * 1.015) verdict = 'Bullish';
    else if (preds.ensemble < avg * 0.985) verdict = 'Bearish';
    setHistory(prev => {
      const latest = {
        coin: selectedCrypto,
        price: formatPrice(preds.ensemble),
        verdict,
      };
      if (prev.length && prev[prev.length - 1].coin === latest.coin && prev[prev.length - 1].price === latest.price) return prev;
      return [...prev.slice(-14), latest];
    });
    setModelHistory(prev => {
      const updated = { ...prev };
      Object.keys(prev).forEach(k => {
        updated[k] = [...(prev[k] || []), prediction.predictions[k as keyof PredictionData["predictions"]]].slice(-20);
      });
      return updated;
    });
  }, [prediction, selectedCrypto]);

  return (
    <div className={dark ? "dark min-h-screen w-full bg-[#191c20]" : "min-h-screen w-full bg-[#f3f4f6]"}>
      {/* THEME TOGGLE BUTTON */}
      <button
        onClick={() => setDark(d => !d)}
        className="fixed top-5 right-5 z-50 bg-white dark:bg-[#20232b] border border-gray-200 dark:border-gray-700 rounded-full shadow px-4 py-2 flex items-center gap-1 text-sm font-semibold select-none transition hover:bg-gray-100 dark:hover:bg-[#232635]"
        aria-label="Toggle Dark Mode"
        style={{ minWidth: 48 }}
      >
        {dark ? "🌙 Dark" : "☀️ Light"}
      </button>
      {/* SCROLLING MARQUEE ticker */}
      <div className="w-full max-w-3xl mx-auto overflow-hidden mb-4"
        style={{
          background: dark ? '#23252b' : '#e0e1e5',
          borderRadius: '15px',
          minHeight: '44px',
          display: 'flex',
          alignItems: 'center',
          border: dark ? '1.5px solid #32353c' : '1.5px solid #ececec'
        }}>
        <div
          ref={marqueeRef}
          className="whitespace-nowrap"
          style={{
            display: 'inline-block',
            animation: 'marquee 15s linear infinite',
            fontWeight: 600,
            fontSize: 18,
            color: dark ? '#eee' : '#111827',
            letterSpacing: '.2px',
            padding: '3px 0'
          }}
        >
          <div className="inline-flex items-center gap-10 px-4">
            {tickerContent}
          </div>
          <span className="mx-8" />
          <div className="inline-flex items-center gap-10 px-4">
            {tickerContent}
          </div>
        </div>
        <style jsx global>{`
          @keyframes marquee {
            0% { transform: translateX(0%);}
            100% { transform: translateX(-50%);}
          }
        `}</style>
      </div>
      <div className="max-w-3xl mx-auto py-8 px-2">
        {/* Header */}
        <div className="w-full flex flex-col items-center mb-6 mt-4">
          <h1 className="text-3xl md:text-5xl font-extrabold text-center text-gray-900 dark:text-gray-100 tracking-tight mb-2">
            <span className="text-purple-500 dark:text-purple-300">Crypto Vertex</span> <span className="text-gray-700 dark:text-gray-200">AI</span>
          </h1>
          <p className="text-base sm:text-lg text-gray-600 dark:text-gray-300 text-center font-medium max-w-2xl">
            Your <span className="text-purple-500 font-semibold dark:text-purple-300">next-generation</span> tool for digital-asset price insight.
            <br className="hidden sm:block" />
            AI-powered <span className="font-semibold">crypto predictions</span> for smarter trading.
          </p>
        </div>
        {/* Selector */}
        <div className="mb-7 flex flex-col md:flex-row md:items-end md:gap-5 gap-4 items-center">
          <div>
            <label className="text-gray-700 dark:text-gray-200 block mb-1" htmlFor="crypto">
              Cryptocurrency
            </label>
            <select
              id="crypto"
              value={selectedCrypto}
              onChange={(e) => setSelectedCrypto(e.target.value)}
              className="w-full bg-white dark:bg-[#23262a] border border-gray-200 dark:border-gray-700 rounded-md p-3 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-400 mt-1 transition-all"
            >
              {cryptoOptions.map(crypto => (
                <option key={crypto} value={crypto}>{crypto}</option>
              ))}
            </select>
          </div>
        </div>
        
        {/* AI Confidence & Verdict */}
        <div className="mb-8 w-full">
          {prediction && (() => {
            const preds = prediction.predictions;
            const avg = (preds.lstm + preds.xgboost + preds.bidirectional_lstm) / 3;
            let verdict = 'Neutral', emoji = '🤔', color = dark ? 'bg-gray-700 border-gray-600' : 'bg-gray-200', comment = 'Wait and watch, market is consolidating.';
            if (preds.ensemble > avg * 1.015) {
              verdict = 'Bullish';
              emoji = '🚀';
              color = dark ? 'bg-green-900 border-green-800' : 'bg-green-100';
              comment = 'AI is confident: expect upward price action!';
            } else if (preds.ensemble < avg * 0.985) {
              verdict = 'Bearish';
              emoji = '🔻';
              color = dark ? 'bg-red-900 border-red-800' : 'bg-red-100';
              comment = 'Caution: AI predicts a correction or downtrend.';
            }
            const confidence = Math.min(100, Math.abs(preds.ensemble - avg) / (avg || 1) * 100);
            return (
              <div className={`w-full px-4 py-4 mb-6 rounded-xl border-2 ${color} flex flex-col items-center shadow`}>
                <div className="flex gap-3 items-center mb-2">
                  <span className="text-3xl">{emoji}</span>
                  <span className="text-xl md:text-2xl font-bold uppercase tracking-widest text-gray-700 dark:text-gray-100">{verdict} PREDICTION</span>
                </div>
                <div className="w-full max-w-xs flex flex-col items-center">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2">
                    <div className={`h-3 rounded-full transition-all duration-500 ${
                      verdict === 'Bullish' ? 'bg-green-400 dark:bg-green-500' : verdict === 'Bearish' ? 'bg-red-400 dark:bg-red-500' : 'bg-gray-400 dark:bg-gray-600'
                    }`} style={{ width: `${Math.max(18, confidence).toFixed(1)}%` }} />
                  </div>
                  <span className="text-sm font-semibold text-gray-700 dark:text-gray-100">{comment}</span>
                </div>
              </div>
            );
          })()}
        </div>
        {/* Predictions FIRST */}
        <div className="mb-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {predictionModels.map((model, idx) => (
              <motion.div
                key={model.key}
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ delay: idx * 0.14, duration: 0.5, type: "spring" }}
                whileHover={{ scale: 1.06, boxShadow: "0 4px 32px -4px #a78bfa33" }}
                className={`
                  flex flex-col items-center justify-center
                  rounded-xl shadow bg-gray-50 dark:bg-[#23262a] py-6 px-2 sm:px-3 relative cursor-pointer
                  border-2 ${model.accent} transition-all duration-200
                `}
                onClick={() => setModelModal({ open: true, key: model.key })}
              >
                <span className="text-3xl mb-2">{model.emoji}</span>
                <span className={`text-base font-semibold mb-1 ${model.accent}`}>{model.label}</span>
                <span className={`text-xl font-extrabold ${model.accent}`}>
                  {isLoading || !prediction ? (
                    <span className="opacity-60 animate-pulse">...</span>
                  ) : (
                    formatPrice(prediction.predictions[model.key as keyof typeof prediction.predictions])
                  )}
                </span>
              </motion.div>
            ))}
          </div>
        </div>
        {/* Chart BELOW predictions */}
        <div className="bg-gray-100 dark:bg-[#20232b] border border-gray-200 dark:border-gray-700 rounded-xl p-6 mb-8 shadow-sm">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4 text-center">Last 90 Days Price</h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" stroke={dark ? "#353a42" : "#d1d5db"} />
              <XAxis dataKey="Date" tick={{ fill: dark ? '#e5e7eb' : '#9ca3af', fontSize: 12 }} />
              <YAxis domain={['dataMin', 'dataMax']} tick={{ fill: dark ? '#e5e7eb' : '#9ca3af', fontSize: 12 }} />
              <Tooltip
                content={({ active, payload, label }) =>
                  active && payload && payload.length ? (
                    <div className="bg-white dark:bg-[#20232b] p-2 rounded border border-gray-200 dark:border-gray-600 shadow">
                      <div className="text-purple-500 font-bold dark:text-purple-300">{label}</div>
                      <div className="text-gray-800 dark:text-gray-200 font-semibold">Price: {formatPrice(payload[0].value)}</div>
                    </div>
                  ) : null
                }
              />
              <Area type="monotone" dataKey="Close" fill={dark ? "#262b37" : "#f3e8ff"} stroke={dark ? "#c4b5fd" : "#a78bfa"} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        {/* News last */}
        <div className="bg-gray-100 dark:bg-[#23262a] border border-blue-100 dark:border-blue-900 rounded-xl py-4 px-6 mb-2 shadow flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-400">Latest {selectedCrypto.replace('-USD', '')} News</h3>
            {newsLoading && <div className="text-sky-500 dark:text-sky-300 animate-pulse text-xs">loading...</div>}
          </div>
          <ul className="space-y-2">
            {news && news.length > 0 ? news.map((item: any, i: number) => (
              <li key={i} className="flex items-start gap-2 transition hover:scale-[1.01]">
                <span className="text-base mt-0.5 text-blue-400 dark:text-blue-200">•</span>
                <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-sm text-blue-900 dark:text-blue-100 hover:text-sky-500 dark:hover:text-sky-400 underline font-medium transition-colors">
                  {item.title?.length > 110 ? item.title.slice(0, 110) + '...' : item.title}
                </a>
              </li>
            )) : (
              <li className="flex items-start gap-2">
                <span className="text-base mt-0.5 text-blue-400 dark:text-blue-200">•</span>
                <span className="text-sm text-slate-500 dark:text-slate-300 font-medium">
                  Example headline: 'Bitcoin price rises after ETF news. Read more on Coindesk.'
                </span>
              </li>
            )}
          </ul>
        </div>
        {/* Prediction History */}
        <div className="bg-gray-50 dark:bg-[#191c20] border border-gray-200 dark:border-gray-700 rounded-xl py-4 px-6 my-8 shadow flex flex-col">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-lg font-bold text-indigo-800 dark:text-indigo-400">Recent Predictions</h3>
            <span className="ml-2 text-sm text-gray-500 dark:text-gray-300">(current session)</span>
          </div>
          {history.length === 0 ? (
            <div className="text-sm text-gray-500 dark:text-gray-300">No predictions yet. Select a coin to see history!</div>
          ) : (
            <ul className="space-y-2 text-base">
              {history.slice().reverse().map((item, i) => (
                <li key={i} className="flex items-center gap-3 px-1">
                  <span className={
                    item.verdict === 'Bullish' ? 'text-green-500 dark:text-green-400' :
                      item.verdict === 'Bearish' ? 'text-red-500 dark:text-red-400' : 'text-gray-400 dark:text-gray-500'
                  }>
                    {item.verdict === 'Bullish' && '🚀'}
                    {item.verdict === 'Bearish' && '🔻'}
                    {item.verdict === 'Neutral' && '🤔'}
                  </span>
                  <span className="font-semibold">{item.coin}</span>
                  <span className="text-gray-600 dark:text-gray-200">Ensemble: <b>{item.price}</b></span>
                  <span className={`font-semibold uppercase ml-3 text-xs tracking-wider ${
                    item.verdict === 'Bullish' ? 'text-green-600 dark:text-green-400' :
                      item.verdict === 'Bearish' ? 'text-red-500 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {item.verdict}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        {/* Model Info Drawer */}
        {modelModal.open && (() => {
          const key = modelModal.key;
          const mh = modelHistory[key] || [];
          const name = predictionModels.find(m => m.key === key)?.label ?? key;
          const emoji = predictionModels.find(m => m.key === key)?.emoji ?? '';
          const desc = {
            ensemble: 'A “blended” AI model using LSTM, Bi-LSTM, and XGBoost for better accuracy.',
            lstm: 'A deep-learning model that predicts crypto prices from sequential data.',
            bidirectional_lstm: 'Like LSTM, but reads time series both forward and backward for stronger patterns.',
            xgboost: 'A gradient-boosted tree model—excels at complex, non-sequential correlations.'
          }[key] || '';
          return (
            <div className="fixed inset-0 z-30 bg-black/30 flex items-center justify-center" onClick={() => setModelModal({ open: false, key: '' })}>
              <div className="bg-white dark:bg-[#22232a] rounded-xl border-2 border-purple-100 dark:border-purple-800 shadow-xl w-full max-w-md mx-auto p-6 relative"
                onClick={e => e.stopPropagation()}>
                <button className="absolute top-2 right-3 text-gray-400 dark:text-gray-300 text-xl font-bold" onClick={() => setModelModal({ open: false, key: '' })}>&times;</button>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-3xl">{emoji}</span>
                  <span className="font-bold text-lg">{name}</span>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-300 mb-2 italic">{desc}</p>
                <div className="font-mono text-[1.03rem] mb-4 text-purple-700 dark:text-purple-300">
                  Current Value: <span className="font-bold">{prediction ? formatPrice(prediction.predictions[key as keyof PredictionData["predictions"]]) : '--'}</span>
                </div>
                <div className="w-full h-20 flex items-end mb-2">
                  {/* Mini chart */}
                  <svg width="100%" height="100%" viewBox={`0 0 140 55`}>
                    <polyline
                      fill="none"
                      stroke={dark ? "#c4b5fd" : "#a78bfa"}
                      strokeWidth="3"
                      points={
                        mh.map((v, i) => {
                          const x = 7 + i * (126 / (mh.length - 1 || 1));
                          const min = Math.min(...mh); const max = Math.max(...mh);
                          const y = 48 - ((v - (min || 0)) / ((max - min) || 1) * 42);
                          return `${x},${y.toFixed(1)}`;
                        }).join(' ')
                      }
                    />
                  </svg>
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-300">Last {mh.length} predictions</div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
