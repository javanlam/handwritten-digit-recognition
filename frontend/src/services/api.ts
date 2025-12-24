import { GridData, PredictionResult } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export class InferenceAPI {
  static async predict(imageArray: number[][]): Promise<PredictionResult> {
    try {
      // Convert 2D array to 1D list for the API
      // const flatArray = imageArray.flat();
      
      const response = await fetch(`${API_BASE_URL}/api/inference`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          img_array: imageArray
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Parse the JSON string returned by FastAPI
      if (typeof data === 'string') {
        return JSON.parse(data);
      }
      
      return data as PredictionResult;
    } catch (error) {
      console.error('Error calling inference API:', error);
      throw error;
    }
  }
}