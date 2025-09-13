'use client';

import React from 'react';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ReactTyped = React as any;
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { BrainCircuit, Cpu, Zap, Gem } from 'lucide-react';

// --- Helper Functions ---

const formatPrice = (price: number | undefined) => {
    if (price === undefined) return '$...';
    return '$' + price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-700/80 backdrop-blur-sm p-3 rounded-lg border border-slate-600">
                <p className="label text-slate-300">{`${label}`}</p>
                <p className="intro text-white font-bold">{`Price: ${formatPrice(payload[0].value)}`}</p>
            </div>
        );
    }
    return null;
};

// --- Animation Variants ---
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 100 }
  }
};

// --- Main Component ---
export default function PredictionDashboard() {
    const [prediction, setPrediction] = ReactTyped.useState(null);
    const [historicalData, setHistoricalData] = ReactTyped.useState([]);
    const [isLoading, setIsLoading] = ReactTyped.useState(true);
    const [error, setError] = ReactTyped.useState(null);
    const [selectedCrypto, setSelectedCrypto] = ReactTyped.useState('BTC-USD');
    const cryptoOptions = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD'];
    const API_BASE_URL = "http://127.0.0.1:8000";

    const modelIcons = {
        lstm: <BrainCircuit size={20} className="text-cyan-400" />,
        bilstm: <Zap size={20} className="text-blue-400" />,
        xgboost: <Cpu size={20} className="text-green-400" />,
    };

    // DEBUGGING VERSION of fetchData
    const fetchData = async () => {
        console.log("1. Starting fetchData function...");
        setIsLoading(true);
        setError(null);

        try {
            console.log("2. Preparing to call backend at:", API_BASE_URL);

            const predResponse = await axios.post(`${API_BASE_URL}/predict`, { symbol: selectedCrypto });
            
            console.log("3. SUCCESS: Received prediction response:", predResponse.data);

            const histResponse = await axios.get(`${API_BASE_URL}/history`, { params: { symbol: selectedCrypto } });

            console.log("4. SUCCESS: Received history response:", histResponse.data);

            setPrediction(predResponse.data);
            setHistoricalData(histResponse.data);
            
            console.log("5. State updated successfully.");

        } catch (err) {
            console.error("!!! CAUGHT AN ERROR:", err);
            setError("Failed to fetch data. Check the console for the error message.");
        } finally {
            console.log("6. Reached the 'finally' block. Setting loading to false.");
            setIsLoading(false);
        }
    };

    ReactTyped.useEffect(() => {
        fetchData();
    }, [selectedCrypto]); // Runs when selectedCrypto changes

    return (
        <div className="min-h-screen text-white">
            <div className="container mx-auto px-4 py-8">
                <motion.header
                    variants={itemVariants}
                    initial="hidden"
                    animate="visible"
                    className="text-center mb-12"
                >
                    <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                        Crypto Vertex AI
                    </h1>
                    <p className="text-slate-400 mt-2">Real-time Price Predictions Powered by Srikar Vaka</p>
                </motion.header>

                <motion.div
                    className="grid grid-cols-1 lg:grid-cols-4 gap-6"
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                >
                    {/* Main Content: Chart + Ensemble */}
                    <motion.div variants={itemVariants} className="lg:col-span-3 flex flex-col gap-6">
                        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl p-6 flex-grow">
                             <h3 className="text-xl font-semibold mb-4">{selectedCrypto} - Last 90 Days Price</h3>
                             <ResponsiveContainer width="100%" height={350}>
                                <AreaChart data={historicalData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                                    <defs>
                                        <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                                            <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis dataKey="Date" stroke="#9CA3AF" tick={{ fontSize: 12 }} interval={14} />
                                    <YAxis stroke="#9CA3AF" tickFormatter={(value: number) => `$${Number(value/1000).toLocaleString()}k`} tick={{ fontSize: 12 }} domain={['dataMin - 1000', 'dataMax + 1000']} />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Area type="monotone" dataKey="Close" stroke="#8884d8" strokeWidth={2} fillOpacity={1} fill="url(#colorClose)" name="Close Price" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>

                    {/* Sidebar: Controls + Predictions */}
                    <motion.div variants={itemVariants} className="lg:col-span-1 flex flex-col gap-6">
                        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl p-6">
                             <h3 className="text-lg font-semibold mb-4">Controls</h3>
                            {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                            <select value={selectedCrypto} onChange={(e: any) => setSelectedCrypto(e.target.value)} className="w-full bg-slate-700 rounded-md p-3 text-white focus:outline-none focus:ring-2 focus:ring-purple-500">
                                {cryptoOptions.map(crypto => (<option key={crypto} value={crypto}>{crypto}</option>))}
                            </select>
                            <button onClick={fetchData} disabled={isLoading} className="w-full mt-4 bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded-md transition-all duration-300 disabled:bg-slate-500 disabled:cursor-not-allowed">
                                {isLoading ? 'Loading...' : 'Get Prediction'}
                            </button>
                        </div>

                        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl p-6">
                           <div className="flex items-center gap-3 mb-4">
                             <Gem size={24} className="text-purple-400" />
                             <h3 className="text-lg font-semibold">Ensemble Prediction</h3>
                           </div>
                           <AnimatePresence mode="wait">
                               <motion.p 
                                 key={prediction ? prediction.predictions.ensemble : 'loading'}
                                 initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }}
                                 className="text-4xl font-bold text-white text-center py-4"
                               >
                                 {isLoading ? '...' : formatPrice(prediction?.predictions.ensemble)}
                               </motion.p>
                           </AnimatePresence>
                        </div>
                        
                        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl p-6">
                            <h3 className="text-lg font-semibold mb-4">Individual Models</h3>
                             <div className="space-y-4 mt-4">
                               <div className="flex justify-between items-center"><span className="flex items-center gap-2 text-slate-400">{modelIcons.lstm} LSTM</span> <span className="font-mono">{isLoading ? '...' : formatPrice(prediction?.predictions.lstm)}</span></div>
                               <div className="flex justify-between items-center"><span className="flex items-center gap-2 text-slate-400">{modelIcons.bilstm} Bi-LSTM</span> <span className="font-mono">{isLoading ? '...' : formatPrice(prediction?.predictions.bidirectional_lstm)}</span></div>
                               <div className="flex justify-between items-center"><span className="flex items-center gap-2 text-slate-400">{modelIcons.xgboost} XGBoost</span> <span className="font-mono">{isLoading ? '...' : formatPrice(prediction?.predictions.xgboost)}</span></div>
                            </div>
                            {error && <p className="text-red-400 text-center mt-4">{error}</p>}
                        </div>
                    </motion.div>
                </motion.div>
            </div>
        </div>
    );
}