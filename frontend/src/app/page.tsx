'use client';

import { useState, useRef, useEffect } from 'react';
import { Upload, Image as ImageIcon, Sparkles, Download, ArrowRight } from 'lucide-react';
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [restoredUrl, setRestoredUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [boostColor, setBoostColor] = useState(true);
  const [sharpnessLevel, setSharpnessLevel] = useState(0);
  const [enhanceFaces, setEnhanceFaces] = useState(true);
  const [colorize, setColorize] = useState(false);
  
  // IP address of the DGX Server where FastAPI is running
  // The user should change "localhost" to their DGX server's public IP or Ngrok URL
  const [apiUrl, setApiUrl] = useState('http://localhost:8000/api/enhance');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setOriginalUrl(URL.createObjectURL(file));
      setRestoredUrl(null);
    }
  };

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
        const file = e.clipboardData.files[0];
        if (file.type.startsWith('image/')) {
          setSelectedFile(file);
          setOriginalUrl(URL.createObjectURL(file));
          setRestoredUrl(null);
        }
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  const handleEnhance = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('boost_color', boostColor.toString());
    formData.append('sharpness_level', (sharpnessLevel / 100).toString());
    formData.append('enhance_faces', enhanceFaces.toString());
    formData.append('colorize', colorize.toString());

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true'
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to enhance image');
      }

      const blob = await response.blob();
      setRestoredUrl(URL.createObjectURL(blob));
    } catch (error) {
      console.error(error);
      alert('Error connecting to AI Backend! Please make sure your DGX server is running uvicorn, and that you have entered the correct API URL (e.g. your Ngrok URL or public IP) in the settings below.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white p-8 font-sans">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 mt-8">
          <h1 className="text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600 mb-4">
            Ultimate Photo Restorer
          </h1>
          <p className="text-gray-400 text-xl">
            Commercial-Grade AI Face Restoration & 4K Upscaling
          </p>
        </div>

        {/* Server Settings */}
        <div className="max-w-2xl mx-auto mb-12 bg-gray-800/30 p-4 rounded-xl border border-gray-700 flex items-center space-x-4">
            <span className="text-gray-400 font-medium whitespace-nowrap">DGX Server URL:</span>
            <input 
              type="text" 
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="flex-1 bg-gray-900 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
              placeholder="http://<your-dgx-ip>:8000/api/enhance"
            />
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-12 gap-8">
          
          {/* Controls Section */}
          <div className="lg:col-span-4 bg-gray-800/50 backdrop-blur-xl border border-gray-700 p-8 rounded-3xl shadow-2xl flex flex-col max-h-[85vh] overflow-y-auto">
            
            <div className="border-2 border-dashed border-gray-600 rounded-2xl p-10 text-center hover:border-blue-500 transition-colors cursor-pointer relative flex flex-col items-center justify-center bg-gray-900/50">
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleFileChange} 
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Upload className="w-12 h-12 mb-4 text-blue-400" />
              <p className="text-lg font-bold">Upload Photo</p>
              <p className="text-gray-500 mt-2 text-sm">Drag & Drop, Click, or press Ctrl+V</p>
            </div>

            {originalUrl && (
              <div className="mt-8 space-y-6">
                <label className="flex items-center space-x-3 cursor-pointer bg-gray-900/50 p-4 rounded-xl border border-gray-700">
                  <input 
                    type="checkbox" 
                    checked={enhanceFaces} 
                    onChange={(e) => setEnhanceFaces(e.target.checked)}
                    className="w-5 h-5 rounded border-gray-600 text-green-500 focus:ring-green-500"
                  />
                  <div>
                    <span className="text-gray-300 font-medium block">Enhance Human Faces</span>
                    <span className="text-gray-500 text-sm">Uncheck for animals, anime, or landscapes</span>
                  </div>
                </label>

                <label className="flex items-center space-x-3 cursor-pointer bg-gray-900/50 p-4 rounded-xl border border-gray-700">
                  <input 
                    type="checkbox" 
                    checked={colorize} 
                    onChange={(e) => setColorize(e.target.checked)}
                    className="w-5 h-5 rounded border-gray-600 text-purple-500 focus:ring-purple-500"
                  />
                  <div>
                    <span className="text-gray-300 font-medium block">Colorize B&W Photo</span>
                    <span className="text-gray-500 text-sm">Fills in vibrant colors automatically</span>
                  </div>
                </label>

                <label className="flex items-center space-x-3 cursor-pointer bg-gray-900/50 p-4 rounded-xl border border-gray-700">
                  <input 
                    type="checkbox" 
                    checked={boostColor} 
                    onChange={(e) => setBoostColor(e.target.checked)}
                    className="w-5 h-5 rounded border-gray-600 text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-gray-300 font-medium">Apply Cinematic Color Boost</span>
                </label>

                <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-700">
                  <label className="flex flex-col space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-300 font-medium">Sharpness Intensity</span>
                      <span className="text-purple-400 font-bold">{sharpnessLevel}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0" 
                      max="100" 
                      value={sharpnessLevel}
                      onChange={(e) => setSharpnessLevel(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-gray-500 text-sm">Slide right to harden soft edges and clarify details</span>
                  </label>
                </div>

                <button 
                  onClick={handleEnhance}
                  disabled={isProcessing}
                  className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center space-x-2 transition-all ${
                    isProcessing 
                      ? 'bg-gray-600 cursor-not-allowed text-gray-300' 
                      : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 shadow-lg shadow-purple-500/25 text-white'
                  }`}
                >
                  {isProcessing ? (
                    <span className="animate-pulse">Enhancing on DGX Server...</span>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5" />
                      <span>Generate 4K Restoration</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Preview Section */}
          <div className="lg:col-span-8 bg-gray-800/50 backdrop-blur-xl border border-gray-700 p-8 rounded-3xl shadow-2xl min-h-[500px] lg:h-[85vh] flex flex-col">
            {!originalUrl ? (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
                <ImageIcon className="w-20 h-20 mb-4 opacity-20" />
                <p className="text-lg">Upload an image to see the interactive preview</p>
              </div>
            ) : restoredUrl ? (
              <div className="flex-1 flex flex-col h-full">
                <div className="flex-1 rounded-xl overflow-hidden shadow-2xl border border-gray-600 relative h-full">
                  <ReactCompareSlider
                    itemOne={<ReactCompareSliderImage src={originalUrl} alt="Original" style={{ objectFit: 'contain' }} />}
                    itemTwo={<ReactCompareSliderImage src={restoredUrl} alt="Restored" style={{ objectFit: 'contain' }} />}
                    position={50}
                    className="h-full bg-black"
                  />
                </div>
                <div className="mt-4 flex justify-between items-center text-gray-400">
                  <span className="font-medium flex items-center">
                    <ArrowRight className="w-4 h-4 mr-2" />
                    Drag slider to compare Before & After
                  </span>
                  <a 
                    href={restoredUrl} 
                    download="restored_4k.jpg"
                    className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 px-6 py-2 rounded-lg text-white font-bold transition-all shadow-lg"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download 4K</span>
                  </a>
                </div>
              </div>
            ) : (
              <div className="flex-1 rounded-xl overflow-hidden shadow-2xl border border-gray-600 flex items-center justify-center relative bg-black">
                <img src={originalUrl} className="max-w-full max-h-full object-contain opacity-40 blur-sm" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="bg-black/80 border border-gray-700 px-8 py-4 rounded-full text-white font-bold shadow-2xl">
                    Ready to Enhance
                  </span>
                </div>
              </div>
            )}
          </div>
          
        </div>
      </div>
    </main>
  );
}
