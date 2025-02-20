'use client'

import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'

export default function TextAnalysis() {
  const [text, setText] = useState('')
  const [result, setResult] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [height, setHeight] = useState('200px')
  const [isCollapsed, setIsCollapsed] = useState(false)
  
  const sourceTextRef = useRef<HTMLTextAreaElement>(null)
  const resultTextRef = useRef<HTMLTextAreaElement>(null)
  
  const MAX_HEIGHT = 600
  const MIN_HEIGHT = 200

  const updateHeight = () => {
    if (sourceTextRef.current && resultTextRef.current) {
      if (isCollapsed) {
        setHeight(`${MIN_HEIGHT}px`)
        return
      }
      
      const sourceHeight = sourceTextRef.current.scrollHeight
      const resultHeight = resultTextRef.current.scrollHeight
      const maxHeight = Math.min(
        Math.max(sourceHeight, resultHeight, MIN_HEIGHT),
        MAX_HEIGHT
      )
      setHeight(`${maxHeight}px`)
    }
  }

  useEffect(() => {
    updateHeight()
  }, [text, result, isCollapsed])

  const analyzeText = async () => {
    if (!text.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze text');
      }

      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      console.error('Error:', error);
      setResult('Произошла ошибка при анализе текста');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
        <h1 className="text-[#2d4858] text-2xl font-medium mb-6 container mx-auto px-4 max-w-6xl">
          Анализ текста
        </h1>
        
        <div className="w-screen relative left-[50%] right-[50%] mx-[-50vw] bg-[#f0f2f5] mb-12">
          <Card className="bg-[#f0f2f5] py-8">
            <div className="container mx-auto px-4 max-w-6xl">
              <div className="shadow-[0_0_20px_rgba(0,0,0,0.2)] bg-white rounded-[20px]">
                <div className="relative">
                  <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    className="absolute -top-6 right-4 text-sm text-gray-500 hover:text-gray-700 transition-colors"
                  >
                    {isCollapsed ? 'Показать' : 'Скрыть'}
                  </button>
                  <div className="grid md:grid-cols-2 gap-0">
                    <div className="border-r border-gray-300">
                      <div className="relative">
                        <Textarea
                          ref={sourceTextRef}
                          placeholder="Исходный текст"
                          className="resize-none border-0 bg-white text-[#2d4858] rounded-r-none rounded-l-[20px] p-4 overflow-y-auto transition-[height]"
                          style={{ 
                            height,
                            maxHeight: `${MAX_HEIGHT}px`
                          }}
                          value={text}
                          onChange={(e) => setText(e.target.value)}
                        />
                        <div className="absolute bottom-6 right-6 text-sm text-gray-500">
                          {text.length} / 10000
                        </div>
                      </div>
                    </div>
                    
                    <div className="relative">
                      <Textarea
                        ref={resultTextRef}
                        placeholder="Результат"
                        className="resize-none border-0 bg-[#f2f4f9] text-[#2d4858] rounded-l-none rounded-r-[20px] p-4 overflow-y-auto transition-[height]"
                        style={{ 
                          height,
                          maxHeight: `${MAX_HEIGHT}px`
                        }}
                        value={result}
                        readOnly
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex justify-center mt-8">
              <Button
                className="bg-white text-[#2d4858] hover:bg-[#E8F0FB]/90 px-8 rounded-[10px] shadow-[0_0_20px_rgba(0,0,0,0.2)]"
                onClick={analyzeText}
                disabled={isLoading}
              >
                {isLoading ? 'Анализируем...' : 'Анализировать текст'}
              </Button>
            </div>
          </Card>
        </div>

        <section className="mb-20 container mx-auto px-4 max-w-6xl">
          <h2 className="text-[#2d4858] text-2xl font-medium mb-4">
            Что умеет сервис?
          </h2>
          <p className="text-[#2d4858] leading-relaxed">
            Блабла блаблабла бла блабла Блабла блаблабла бла блабла Блабла блаблабла бла блабла Блабла блаблабла бла блабла!!!
            Блабла блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блабла
            блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блабла!!!
            Блабла блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блабла
            блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блаблаБлабла блаблабла бла блабла.
          </p>
        </section>
      </main>

      <footer className="bg-[#dee3eb] py-4">
        <div className="container mx-auto px-4 max-w-6xl flex justify-between items-center text-sm text-[#2d4858]">
          <div className="space-y-1">
            <div>Contact Us</div>
            <div>example_name@mail.com</div>
          </div>
          <div className="text-right space-y-1">
            <div>Saint-Petersburg, 197101, Russia</div>
            <div>© 2025</div>
          </div>
        </div>
      </footer>
    </div>
  )
}
