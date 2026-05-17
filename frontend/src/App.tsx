import { useState, useEffect } from 'react';
import './index.css';
import AssessmentForm from './components/AssessmentForm';
import AssessmentList from './components/AssessmentList';
import type { AssessmentResult } from './components/AssessmentList';

interface ApiResponse {
  health_risk_level: string;
  audio_analysis: AssessmentResult['audio_analysis'];
  status: string;
}

function App() {
  const [results, setResults] = useState<AssessmentResult[]>(() => {
    const saved = localStorage.getItem('pregnancy_assessments');
    return saved ? JSON.parse(saved) : [];
  });
  const [isLoading, setIsLoading] = useState(false);

  // Save to localStorage
  useEffect(() => {
    localStorage.setItem('pregnancy_assessments', JSON.stringify(results));
  }, [results]);

  const handleSubmit = async (formData: FormData) => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Falha na comunicação com o servidor');

      const data: ApiResponse = await response.json();
      
      const newResult: AssessmentResult = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        health_risk_level: data.health_risk_level,
        audio_analysis: data.audio_analysis,
      };

      setResults((prev) => [newResult, ...prev]);
    } catch (error) {
      console.error(error);
      alert('Erro ao processar avaliação. Verifique se o backend está rodando.');
    } finally {
      setIsLoading(false);
    }
  };

  const clearResults = () => {
    if (window.confirm('Tem certeza que deseja limpar todo o histórico?')) {
      setResults([]);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>🤰 Pregnancy Health Helper</h1>
        <p style={{textAlign: 'center', color: '#666'}}>Auxílio no diagnóstico e monitoramento da saúde gestacional</p>
      </header>

      <main>
        <AssessmentForm onSubmit={handleSubmit} isLoading={isLoading} />
        
        {isLoading && (
          <div className="loading-spinner">
            Analisando biomarcadores e áudio...
          </div>
        )}

        <AssessmentList results={results} onClear={clearResults} />
      </main>
    </div>
  );
}

export default App;
