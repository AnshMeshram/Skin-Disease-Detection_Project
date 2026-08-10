import { create } from 'zustand';

export const useDiagnosticStore = create((set) => ({
  // State
  apiStatus: 'offline',
  patientInfo: {
    name: '',
    gender: '',
    age: '',
    site: '',
  },
  currentFile: null,
  previewUrl: null,
  predictionResult: null,
  isAnalyzing: false,

  // Actions
  setApiStatus: (status) => set({ apiStatus: status }),
  setPatientInfo: (info) =>
    set((state) => ({
      patientInfo: typeof info === 'function' ? info(state.patientInfo) : { ...state.patientInfo, ...info },
    })),
  setFileAndPreview: (file, previewUrl) => set({ currentFile: file, previewUrl }),
  setPredictionResult: (result) => set({ predictionResult: result }),
  setIsAnalyzing: (status) => set({ isAnalyzing: status }),
  resetSession: () =>
    set({
      patientInfo: { name: '', gender: '', age: '', site: '' },
      currentFile: null,
      previewUrl: null,
      predictionResult: null,
      isAnalyzing: false,
    }),
}));
