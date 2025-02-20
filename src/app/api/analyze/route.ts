import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(req: Request) {
  try {
    const { text } = await req.json();

    const scriptPath = path.join(process.cwd(), 'text_analyzer', 'text_analyzer.py');
    
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python3', [scriptPath]);
      let result = '';
      let error = '';

      pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        reject(NextResponse.json({ 
          error: 'Failed to start Python process. Please ensure Python 3 is installed and accessible.' 
        }, { status: 500 }));
      });

      pythonProcess.stdin.write(text);
      pythonProcess.stdin.end();

      pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        error += data.toString();
        console.error('Python stderr:', data.toString());
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Python process exited with code:', code);
          console.error('Error output:', error);
          reject(NextResponse.json({ 
            error: `Process exited with code ${code}: ${error}` 
          }, { status: 500 }));
        } else {
          resolve(NextResponse.json({ result }));
        }
      });
    });
  } catch (error) {
    console.error('API route error:', error);
    return NextResponse.json({ 
      error: 'Failed to analyze text: ' + (error as Error).message 
    }, { status: 500 });
  }
} 