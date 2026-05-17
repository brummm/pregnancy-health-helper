import React, { useState } from 'react';

interface AssessmentFormProps {
  onSubmit: (formData: FormData) => void;
  isLoading: boolean;
}

const AssessmentForm: React.FC<AssessmentFormProps> = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    age: '',
    systolic_bp: '',
    diastolic_bp: '',
    bs: '',
    heart_rate: '',
    body_temp: '',
  });
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [transcriptionMethod, setTranscriptionMethod] = useState<'local' | 'gemini'>('local');
  const [bypassTranscription, setBypassTranscription] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setAudioFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const data = new FormData();
    Object.entries(formData).forEach(([key, value]) => data.append(key, value));
    if (audioFile) data.append('audio', audioFile);
    if (bypassTranscription) data.append('bypass_transcription', 'true');
    data.append('transcription_method', transcriptionMethod);
    onSubmit(data);
    setBypassTranscription(false);
  };

  const fillSampleData = () => {
    setFormData({
      age: '25',
      systolic_bp: '130',
      diastolic_bp: '80',
      bs: '15',
      heart_rate: '86',
      body_temp: '36.6',
    });
  };

  const fillSampleDataWithTranscription = () => {
    fillSampleData();
    setBypassTranscription(true);
  };

  return (
    <>
      <div className="dev-controls">
        <h4>Dev Controls</h4>
        <button type="button" className="btn-secondary" style={{ width: 'auto', marginBottom: 0 }} onClick={fillSampleData}>
          🧪 Preencher Exemplo
        </button>
        <button type="button" className="btn-secondary" style={{ width: 'auto', marginBottom: 0 }} onClick={fillSampleDataWithTranscription}>
          🧪 Exemplo + Bypass Transcription
        </button>
      </div>

      <div className="card">
        <h2 style={{ marginBottom: '20px' }}>Nova Avaliação</h2>
        <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Idade (9-80)</label>
          <input type="number" name="age" min="9" max="80" value={formData.age} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Pressão Sistólica (30-200)</label>
          <input type="number" name="systolic_bp" min="30" max="200" value={formData.systolic_bp} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Pressão Diastólica (30-200)</label>
          <input type="number" name="diastolic_bp" min="30" max="200" value={formData.diastolic_bp} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Glicose (BS) (3-30)</label>
          <input type="number" step="0.1" name="bs" min="3" max="30" value={formData.bs} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Frequência Cardíaca (50-150)</label>
          <input type="number" name="heart_rate" min="50" max="150" value={formData.heart_rate} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Temperatura Corporal (°C)</label>
          <input type="number" step="0.1" name="body_temp" min="30" max="45" value={formData.body_temp} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Áudio da Entrevista (Português)</label>
          <input type="file" accept="audio/*" onChange={handleFileChange} />
        </div>

        <div className="form-group">
          <label>Método de Transcrição</label>
          <div style={{ display: 'flex', gap: '20px', marginTop: '5px' }}>
            <label style={{ fontWeight: 'normal', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <input 
                type="radio" 
                name="transcriptionMethod" 
                value="local" 
                checked={transcriptionMethod === 'local'} 
                onChange={() => setTranscriptionMethod('local')} 
              />
              Local (Gratuito/CPU)
            </label>
            <label style={{ fontWeight: 'normal', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <input 
                type="radio" 
                name="transcriptionMethod" 
                value="gemini" 
                checked={transcriptionMethod === 'gemini'} 
                onChange={() => setTranscriptionMethod('gemini')} 
              />
              Gemini API (Alta Precisão/Grátis)
            </label>
          </div>
        </div>

        <button type="submit" disabled={isLoading}>
          {isLoading ? (
            <>
              <span className="spinner"></span>
              Analisando...
            </>
          ) : 'Realizar Diagnóstico'}
        </button>
      </form>
    </div>
    </>
  );
};

export default AssessmentForm;
