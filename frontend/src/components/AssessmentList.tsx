import React, { useState } from 'react';

export interface Condition {
    name: string;
    matches?: string[];
}

export interface AssessmentResult {
  id: string;
  timestamp: string;
  health_risk_level: string;
  audio_analysis?: {
    transcription: string;
    overall_sentiment: string;
    acoustic_metrics?: {
        mean_pitch: number;
        pitch_variability: number;
        mean_energy: number;
        scratchiness_index: number;
        speech_duration_ratio: number;
    };
    detected_conditions: (string | Condition)[];
  } | null;
}

interface AssessmentListProps {
  results: AssessmentResult[];
  onClear: () => void;
}

const AssessmentList: React.FC<AssessmentListProps> = ({ results, onClear }) => {
  const [isVisible, setIsVisible] = useState(true);
  const [expandedTranscriptions, setExpandedTranscriptions] = useState<Record<string, boolean>>({});

  const toggleTranscription = (id: string) => {
    setExpandedTranscriptions(prev => ({ ...prev, [id]: !prev[id] }));
  };

  if (results.length === 0) return null;

  return (
    <div className="card">
      <div className="results-header">
        <h2>Histórico de Avaliações</h2>
        <div>
          <button className="icon-btn" onClick={() => setIsVisible(!isVisible)} title={isVisible ? "Ocultar" : "Mostrar"}>
            {isVisible ? '👁️' : '🙈'}
          </button>
          <button className="icon-btn" onClick={onClear} title="Limpar tudo">
            🗑️
          </button>
        </div>
      </div>

      {isVisible && (
        <div className="assessment-list">
          {results.map((res) => (
            <div key={res.id} className="assessment-item">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <small>{new Date(res.timestamp).toLocaleString()}</small>
                <span className={`risk-tag risk-${res.health_risk_level.split(' ')[0]}`}>
                  {res.health_risk_level}
                </span>
              </div>
              
              {res.audio_analysis && (
                <div className="audio-analysis">
                  <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginBottom: '10px', alignItems: 'center' }}>
                    <span><strong>Emoção:</strong> {res.audio_analysis.overall_sentiment}</span>
                    {res.audio_analysis.acoustic_metrics && (
                        <div style={{ fontSize: '11px', color: '#666', background: '#eee', padding: '2px 8px', borderRadius: '10px' }}>
                            Acoustics: {res.audio_analysis.acoustic_metrics.mean_pitch}Hz | 
                            Var: {res.audio_analysis.acoustic_metrics.pitch_variability} | 
                            Rate: {Math.round(res.audio_analysis.acoustic_metrics.speech_duration_ratio * 100)}%
                        </div>
                    )}
                  </div>
                  
                  <ul className="condition-list">
                    {res.audio_analysis.detected_conditions.map((cond, idx) => {
                      const isNewFormat = typeof cond !== 'string';
                      const name = isNewFormat ? cond.name : cond;

                      return (
                        <li key={idx} className="condition-item">
                          <span style={{ fontSize: '12px', color: '#888', marginRight: '5px' }}>•</span>
                          <span>{name}</span>
                        </li>
                      );
                    })}
                  </ul>

                  <span 
                    className="toggle-link" 
                    onClick={() => toggleTranscription(res.id)}
                  >
                    {expandedTranscriptions[res.id] ? 'Ocultar Transcrição' : 'Ver Transcrição'}
                  </span>

                  {expandedTranscriptions[res.id] && (
                    <div className="transcription-text">
                      {res.audio_analysis.transcription}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AssessmentList;
