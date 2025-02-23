'use client'

import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"

export default function TextAnalysis() {
  const [text, setText] = useState('')
  const [result, setResult] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [height, setHeight] = useState('200px')
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [mode, setMode] = useState('standard')
  const [bm25Query, setBm25Query] = useState('')
  
  const sourceTextRef = useRef<HTMLTextAreaElement>(null)
  const resultTextRef = useRef<HTMLDivElement>(null)
  
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
        body: JSON.stringify({ 
          text,
          mode,
          query: mode === 'bm25' ? bm25Query : ''
        }),
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
                      <div
                        ref={resultTextRef}
                        className="resize-none border-0 bg-[#f2f4f9] text-[#2d4858] rounded-l-none rounded-r-[20px] p-4 overflow-y-auto transition-[height] whitespace-pre-wrap"
                        style={{ 
                          height,
                          maxHeight: `${MAX_HEIGHT}px`
                        }}
                        dangerouslySetInnerHTML={{ __html: result || 'Результат' }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col items-center mt-4 mb-4 space-y-4">
              <RadioGroup
                defaultValue="standard"
                value={mode}
                onValueChange={(value) => {
                  setMode(value);
                  if (value === 'standard') setBm25Query('');
                }}
                className="flex space-x-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="standard" id="standard" />
                  <Label htmlFor="standard">Стандартный анализ</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="bm25" id="bm25" />
                  <Label htmlFor="bm25">BM25 анализ</Label>
                </div>
              </RadioGroup>

              {mode === 'bm25' && (
                <div className="w-full max-w-xl">
                  <Textarea
                    placeholder="Введите предложение для поиска похожих"
                    className="resize-none border bg-white text-[#2d4858] rounded-[10px] p-4"
                    value={bm25Query}
                    onChange={(e) => setBm25Query(e.target.value)}
                    rows={2}
                  />
                </div>
              )}
            </div>

            <div className="flex justify-center mt-8">
              <Button
                className="bg-white text-[#2d4858] hover:bg-[#E8F0FB]/90 px-8 rounded-[10px] shadow-[0_0_20px_rgba(0,0,0,0.2)]"
                onClick={analyzeText}
                disabled={isLoading || (mode === 'bm25' && !bm25Query.trim())}
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
          <div className="text-[#2d4858] leading-relaxed space-y-6">
            <p>
              Наш сервис позволяет проводить комплексный анализ текста на русском языке, оценивая его сложность по международным шкалам, вычисляя лингвистические характеристики и ключевые метрики удобочитаемости. Он может быть полезен преподавателям, студентам, исследователям и разработчикам образовательных материалов.
            </p>

            <div>
              <h3 className="font-medium mb-2">Уровень CEFR и ACTFL</h3>
              <p>
                Наш сервис автоматически определяет уровень сложности текста по шкалам CEFR и ACTFL. Анализ осуществляется на основе данных, полученных через API из внешних источников.
              </p>
              <p className="mt-2">
                Шкала CEFR оценивает тексты с точки зрения их соответствия уровням владения языком, начиная от базового (A1) и заканчивая профессиональным (C2). Шкала ACTFL позволяет дополнительно классифицировать тексты по уровню, используя таблицу соответствий с CEFR.
              </p>
            </div>

            <div>
              <h3 className="font-medium mb-2">Лингвистические характеристики текста</h3>
              <ul className="list-disc pl-5 space-y-1">
                <li>Длина текста в словах, знаках и предложениях</li>
                <li>Средняя длина слова и предложения</li>
                <li>Коэффициент лексического разнообразия, показывающий, насколько разнообразна лексика в тексте</li>
                <li>Поиск ключевых слов с учетом их значимости в тексте</li>
                <li>Поиск полезных слов, которые могут быть важны для изучения и понимания текста</li>
                <li>Процент слов уровня A1 – C2, что позволяет определить, насколько текст подходит для учащихся определенного уровня</li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium mb-2">Показатели удобочитаемости</h3>
              <p>Для оценки сложности восприятия текста наш сервис рассчитывает следующие индексы:</p>
              <ul className="list-disc pl-5 mt-2 space-y-1">
                <li>Индекс Флеша-Кинкейда – показатель читаемости, учитывающий длину слов и предложений</li>
                <li>Индекс LIX – показатель сложности текста, популярный в скандинавских странах</li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium mb-2">Дополнительные аналитические функции</h3>
              <ul className="list-disc pl-5 space-y-1">
                <li>Подсчет времени на чтение для разных уровней владения языком</li>
                <li>BM25 – оценка релевантности текста на основе его статистических характеристик</li>
                <li>Подсчет статистики по частям речи – анализ распределения существительных, глаголов, прилагательных и других частей речи</li>
              </ul>
            </div>
          </div>
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
