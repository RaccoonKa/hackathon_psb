import { useState, useRef, useEffect, type MouseEvent } from 'react';
import axios from 'axios';
import {
  IconPlus, IconSearch, IconBell, IconMessage, IconMore,
  IconSend, IconFileText,
  IconX, IconCheckCheck, IconBot
} from './Icons';

import PsbLogo from './assets/logo.png';

// --- ИНТЕРФЕЙСЫ ---
export interface SourceCitation {
  source_id?: string | null;
  document_id: string;
  version_id?: string | null;
  document_title: string;
  provision_id?: string | null;
  structure_path: string[];
  excerpt: string;
  page?: number | null;
  char_start?: number | null;
  char_end?: number | null;
  score?: number | null;
}

interface BackendChat {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

interface BackendMessage {
  id: string;
  chat_id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: SourceCitation[];
  trace?: any[];
  confidence?: string | null;
  assumptions?: string[];
  missing_info?: string[];
  refusal_reason?: string | null;
  intent: string | null;
  refused: boolean;
  clarification_needed: boolean;
  created_at: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  time: string;
  text: string;
  sources?: SourceCitation[];
  refused?: boolean;
}

export default function App() {
  // Состояния
  const [chats, setChats] = useState<BackendChat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [searchQuery, setSearchQuery] = useState("");

  const [isRightPanelOpen, setIsRightPanelOpen] = useState(true);
  const [activeChatMenu, setActiveChatMenu] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'excerpt' | 'full'>('excerpt');
  const [inputText, setInputText] = useState("");

  const [isSending, setIsSending] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);

  const [activeDocument, setActiveDocument] = useState<{
    document_id: string;
    title: string;
    filename: string;
    content: string;
    char_count: number;
  } | null>(null);

  const [documentCount, setDocumentCount] = useState<number | null>(null);
  const [isDocumentLoading, setIsDocumentLoading] = useState(false);
  const [documentError, setDocumentError] = useState<string | null>(null);

  const initializationStarted = useRef(false);
  const messageRequestId = useRef(0);
  const documentRequestId = useRef(0);
  const activeChatIdRef = useRef<string | null>(activeChatId);

  useEffect(() => {
    activeChatIdRef.current = activeChatId;
  }, [activeChatId]);

  const [activeExcerpt, setActiveExcerpt] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const loadingSteps = [
    "Поиск в базе знаний...",
    "Сопоставление статей...",
    "Проверка достоверности...",
    "Формирование ответа..."
  ];

  // --- ИНИЦИАЛИЗАЦИЯ ---
  useEffect(() => {
    // React StrictMode запускает эффекты повторно в dev-режиме.
    // Защита не позволяет создать дублирующие чаты.
    if (initializationStarted.current) return;
    initializationStarted.current = true;

    const init = async () => {
      try {
        const [chatsResult, documentsResult] = await Promise.allSettled([
          axios.get('/api/chats'),
          axios.get('/api/documents')
        ]);

        if (chatsResult.status === 'rejected') {
          throw chatsResult.reason;
        }

        const loadedChats: BackendChat[] = chatsResult.value.data.items ?? [];
        setChats(loadedChats);

        if (documentsResult.status === 'fulfilled') {
          const documentsData = documentsResult.value.data;
          setDocumentCount(
            Number(documentsData.total ?? documentsData.items?.length ?? 0)
          );
        } else {
          console.error(
            "Ошибка загрузки количества документов",
            documentsResult.reason
          );
        }

        if (loadedChats.length > 0) {
          activeChatIdRef.current = loadedChats[0].id;
          setActiveChatId(loadedChats[0].id);
        } else {
          await handleCreateChat();
        }
      } catch (error) {
        console.error("Ошибка инициализации приложения", error);
      }
    };

    void init();
  }, []);

  useEffect(() => {
    if (activeChatId) {
      fetchMessages(activeChatId);
    } else {
      setMessages([]);
    }
  }, [activeChatId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSending]);

  // --- API ЗАПРОСЫ ---
  const handleSearch = async (query: string) => {
    setSearchQuery(query);

    try {
      const res = await axios.get('/api/chats', {
        params: { q: query }
      });

      setChats(res.data.items);
    } catch (error) {
      console.error("Ошибка поиска", error);
    }
  };

  const handleCreateChat = async () => {
    try {
      const res = await axios.post('/api/chats', {});

      setChats(prev => [res.data, ...prev]);

      activeChatIdRef.current = res.data.id;
      setActiveChatId(res.data.id);
      setSearchQuery("");
    } catch (error) {
      console.error("Ошибка создания чата", error);
    }
  };

  const handleDeleteChat = async (chatId: string, e: MouseEvent) => {
    e.stopPropagation();

    try {
      await axios.delete(`/api/chats/${chatId}`);

      setChats(prev => prev.filter(c => c.id !== chatId));

      if (activeChatIdRef.current === chatId) {
        activeChatIdRef.current = null;
        setActiveChatId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error("Ошибка удаления", error);
    }

    setActiveChatMenu(null);
  };

  const fetchMessages = async (chatId: string) => {
    const requestId = ++messageRequestId.current;

    setMessages([]);

    try {
      const res = await axios.get(`/api/chats/${chatId}/messages`);

      if (requestId !== messageRequestId.current) return;

      const adapted = (res.data.messages ?? []).map(
        (msg: BackendMessage) => ({
          id: msg.id,
          role: msg.role,
          time: formatTime(msg.created_at),
          text: msg.content,
          sources: msg.sources || [],
          refused: msg.refused
        })
      );

      setMessages(adapted);
    } catch (error) {
      if (requestId === messageRequestId.current) {
        console.error("Ошибка загрузки сообщений", error);
      }
    }
  };

  // --- ОТПРАВКА СООБЩЕНИЯ ---
  const handleSend = async () => {
    if (!inputText.trim() || !activeChatId || isSending) return;

    const currentInput = inputText.trim();
    const chatId = activeChatId;

    setInputText("");
    setIsSending(true);
    setLoadingStep(0);

    const stepInterval = setInterval(() => {
      setLoadingStep(prev => (prev < 3 ? prev + 1 : prev));
    }, 2000);

    try {
      const res = await axios.post(
        `/api/chats/${chatId}/messages`,
        {
          content: currentInput
        }
      );

      clearInterval(stepInterval);

      const userMsg: BackendMessage = res.data.user_message;
      const aiMsg: BackendMessage = res.data.assistant_message;

      if (activeChatIdRef.current === chatId) {
        setMessages(prev => [
          ...prev,
          {
            id: userMsg.id,
            role: 'user',
            time: formatTime(userMsg.created_at),
            text: userMsg.content
          },
          {
            id: aiMsg.id,
            role: 'assistant',
            time: formatTime(aiMsg.created_at),
            text: aiMsg.content,
            sources: aiMsg.sources || [],
            refused: aiMsg.refused
          }
        ]);
      }

      const chatsRes = await axios.get('/api/chats', {
        params: { q: searchQuery }
      });

      setChats(chatsRes.data.items);

    } catch (error: any) {
      clearInterval(stepInterval);

      console.error("Ошибка отправки", error);

      let errorMsg = "Ошибка подключения к ИИ.";

      if (error.response?.status === 503) {
        errorMsg = "Сервис ИИ не настроен.";
      }

      if (error.response?.status === 502) {
        errorMsg = "Ошибка на стороне ИИ (лимит или таймаут).";
      }

      if (error.response?.status === 504) {
        errorMsg = "Превышено время ожидания ответа.";
      }

      if (activeChatIdRef.current === chatId) {
        setMessages(prev => [
          ...prev,
          {
            id: Date.now().toString(),
            role: 'assistant',
            time: formatTime(new Date().toISOString()),
            text: `⚠ ${errorMsg}`,
            refused: true
          }
        ]);
      }
    } finally {
      setIsSending(false);
      setLoadingStep(0);
    }
  };

  // --- ОТКРЫТИЕ ДОКУМЕНТА В ПРАВОЙ ПАНЕЛИ ---
  const openSource = async (docId?: string, excerpt?: string) => {
    if (!docId) return;

    setIsRightPanelOpen(true);
    setActiveTab('excerpt');
    setActiveExcerpt(excerpt || null);
    setDocumentError(null);

    // Если документ уже открыт, меняем только выбранную выдержку.
    if (activeDocument?.document_id === docId) return;

    // Не оставляем на экране предыдущий документ во время загрузки нового.
    const requestId = ++documentRequestId.current;

    setActiveDocument(null);
    setIsDocumentLoading(true);

    try {
      const res = await axios.get(
        `/api/documents/${encodeURIComponent(docId)}`
      );

      if (requestId !== documentRequestId.current) return;

      setActiveDocument(res.data);
    } catch (error) {
      if (requestId !== documentRequestId.current) return;

      console.error("Не удалось загрузить текст документа", error);

      setDocumentError(
        "Не удалось открыть документ. Попробуйте нажать на источник ещё раз."
      );
    } finally {
      if (requestId === documentRequestId.current) {
        setIsDocumentLoading(false);
      }
    }
  };

  const formatTime = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleTimeString('ru-RU', {
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return '14:32';
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#F8F9FA] font-sans text-gray-800 overflow-hidden">

      {/* ШАПКА */}
      <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 shrink-0 z-20">
        <div className="flex items-center gap-4">
          <img
            src={PsbLogo}
            alt="ПСБ"
            className="h-9 object-contain"
          />
        </div>

        <div className="flex items-center gap-4 text-[#0b1e4b]">
          <button className="p-2 hover:bg-gray-100 rounded-full text-gray-500">
            <IconBell className="w-5 h-5" />
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ЛЕВАЯ ПАНЕЛЬ */}
        <aside className="w-[280px] bg-[#0b1e4b] flex flex-col shrink-0 text-white">

          <div className="p-4">
            <button
              onClick={handleCreateChat}
              className="w-full bg-[#f95c19] hover:bg-[#e04a10] text-white py-2.5 px-4 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors shadow-sm"
            >
              <IconPlus className="w-4 h-4" />
              Новый чат
            </button>
          </div>

          <div className="px-4 pb-2">
            <div className="relative">
              <IconSearch className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />

              <input
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                placeholder="Поиск по чатам..."
                className="w-full bg-[#152f60] text-sm text-white placeholder-gray-400 rounded-md py-2 pl-9 pr-3 outline-none focus:ring-1 focus:ring-[#f95c19]"
              />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-2 mt-2 space-y-1">

            <p className="px-2 text-xs font-semibold text-gray-400 mb-1 uppercase tracking-wider">
              История чатов
            </p>

            <p className="px-2 pb-2 text-xs text-gray-400">
              Документов в базе: {
                documentCount === null
                  ? '—'
                  : documentCount.toLocaleString('ru-RU')
              }
            </p>

            {chats.map((chat) => (
              <div
                key={chat.id}
                onClick={() => {
                  activeChatIdRef.current = chat.id;
                  setActiveChatId(chat.id);
                }}
                className={`group relative flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer text-sm transition-colors ${
                  activeChatId === chat.id
                    ? 'bg-[#152f60] border-l-2 border-[#f95c19]'
                    : 'hover:bg-[#152f60]'
                }`}
                onMouseLeave={() => setActiveChatMenu(null)}
              >
                <div className="flex items-center gap-2 overflow-hidden whitespace-nowrap">
                  <IconMessage
                    className={`w-4 h-4 shrink-0 ${
                      activeChatId === chat.id
                        ? 'text-[#f95c19]'
                        : 'text-gray-400'
                    }`}
                  />

                  <span
                    className={`truncate ${
                      activeChatId === chat.id
                        ? 'text-white'
                        : 'text-gray-300'
                    }`}
                  >
                    {chat.title}
                  </span>
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setActiveChatMenu(
                      activeChatMenu === chat.id ? null : chat.id
                    );
                  }}
                  className={`p-1 rounded hover:bg-[#1e3f7a] text-gray-400 hover:text-white transition-opacity ${
                    activeChatMenu === chat.id || activeChatId === chat.id
                      ? 'opacity-100'
                      : 'opacity-0 group-hover:opacity-100'
                  }`}
                >
                  <IconMore className="w-4 h-4" />
                </button>

                {activeChatMenu === chat.id && (
                  <div className="absolute right-0 top-10 w-40 bg-white rounded-md shadow-lg border border-gray-100 py-1 z-50 text-gray-800">
                    <button
                      onClick={(e) => handleDeleteChat(chat.id, e)}
                      className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-red-600"
                    >
                      Удалить
                    </button>
                  </div>
                )}
              </div>
            ))}

            {chats.length === 0 && searchQuery && (
              <div className="text-xs text-gray-500 text-center mt-4">
                Ничего не найдено
              </div>
            )}

          </div>
        </aside>

        {/* ЦЕНТРАЛЬНАЯ ЧАСТЬ */}
        <main className="flex-1 flex flex-col min-w-0 bg-[#F8F9FA]">

          <div className="h-14 px-6 flex items-center border-b border-gray-200 bg-[#F8F9FA] shrink-0">
            <h2 className="font-semibold text-lg text-[#0b1e4b] flex items-center gap-2">
              Юридический помощник

              <span className="bg-[#E0E7FF] text-[#3730A3] text-xs px-2 py-0.5 rounded-full font-bold">
                AI
              </span>
            </h2>
          </div>

          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-4 max-w-[85%] ${
                  msg.role === 'user'
                    ? 'self-end flex-row-reverse'
                    : 'self-start'
                }`}
              >

                <div className="shrink-0 mt-1">
                  {msg.role === 'user' ? (
                    <div className="w-8 h-8 rounded-full bg-[#0b1e4b] text-white flex items-center justify-center font-bold text-xs">
                      ВЫ
                    </div>
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center shadow-sm overflow-hidden p-1">
                      <IconBot className="w-5 h-5 text-[#f95c19]" />
                    </div>
                  )}
                </div>

                <div className={`flex flex-col ${
                  msg.role === 'user' ? 'items-end' : 'items-start'
                }`}>

                  <div className={`p-4 rounded-2xl ${
                    msg.role === 'user'
                      ? 'bg-[#D6E4FF] text-[#0b1e4b] rounded-tr-sm'
                      : 'bg-white border border-gray-200 text-gray-800 shadow-sm rounded-tl-sm'
                  } ${
                    msg.refused ? 'border-orange-200 bg-orange-50' : ''
                  }`}>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                      {msg.text}
                    </p>
                  </div>

                  {/* ЧИПЫ ИСТОЧНИКОВ */}
                  {msg.role === 'assistant' &&
                    msg.sources &&
                    msg.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-2 max-w-full">

                      {msg.sources.map((src, idx) => (
                        <button
                          key={`${msg.id}-${src.source_id || src.document_id}-${src.provision_id || idx}-${idx}`}
                          onClick={() => openSource(
                            src.document_id,
                            src.excerpt
                          )}
                          className="text-xs bg-[#eef2f6] hover:bg-[#e2e8f0] text-[#0b1e4b] px-2.5 py-1.5 rounded-lg border border-gray-200 flex items-center gap-1.5 transition-colors cursor-pointer text-left"
                          title={src.document_title}
                        >
                          <span className="font-bold text-[#f95c19] shrink-0">
                            [{idx + 1}]
                          </span>

                          <span className="truncate max-w-[200px]">
                            {src.document_title}
                          </span>

                          {src.provision_id && (
                            <span className="text-gray-500 font-medium shrink-0">
                              ({src.provision_id})
                            </span>
                          )}

                          {src.page && (
                            <span className="text-gray-400 text-[10px] shrink-0">
                              стр. {src.page}
                            </span>
                          )}
                        </button>
                      ))}

                    </div>
                  )}

                  <div className="flex items-center gap-3 mt-2 text-xs text-gray-500 px-1 w-full">
                    <span className="flex items-center gap-1 ml-auto">
                      {msg.time}

                      {msg.role === 'user' && (
                        <IconCheckCheck className="w-3 h-3 text-blue-500" />
                      )}
                    </span>
                  </div>

                </div>
              </div>
            ))}

            {/* ИНДИКАТОР ЗАГРУЗКИ */}
            {isSending && (
              <div className="flex gap-4 max-w-[85%] self-start">

                <div className="shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center shadow-sm overflow-hidden p-1">
                    <IconBot className="w-5 h-5 text-[#f95c19]" />
                  </div>
                </div>

                <div className="p-4 rounded-2xl bg-white border border-gray-200 shadow-sm rounded-tl-sm flex flex-col gap-2 min-w-[250px]">

                  {loadingSteps.map((step, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center gap-2 text-sm transition-opacity duration-500 ${
                        idx > loadingStep ? 'opacity-30' : 'opacity-100'
                      }`}
                    >
                      {idx < loadingStep ? (
                        <span className="text-green-500 text-xs w-3">
                          ✓
                        </span>
                      ) : idx === loadingStep ? (
                        <span className="w-3 h-3 border-2 border-[#f95c19] border-t-transparent rounded-full animate-spin shrink-0"></span>
                      ) : (
                        <span className="w-3"></span>
                      )}

                      <span className={
                        idx === loadingStep
                          ? 'text-[#0b1e4b] font-medium'
                          : 'text-gray-500'
                      }>
                        {step}
                      </span>
                    </div>
                  ))}

                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 bg-[#F8F9FA] border-t border-gray-200">
            <div className="max-w-4xl mx-auto bg-white border border-gray-300 rounded-2xl flex items-center p-2 shadow-sm focus-within:border-[#0b1e4b] focus-within:ring-1 focus-within:ring-[#0b1e4b] transition-all">

              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                disabled={isSending}
                placeholder="Задайте свой вопрос..."
                className="flex-1 bg-transparent outline-none px-2 text-sm text-gray-700 placeholder-gray-400 disabled:opacity-50"
              />

              <button
                onClick={handleSend}
                disabled={isSending || !inputText.trim()}
                className="bg-[#f95c19] hover:bg-[#e04a10] disabled:bg-gray-300 text-white p-2 rounded-xl transition-colors ml-2 flex items-center justify-center"
              >
                <IconSend className="w-4 h-4" />
              </button>

            </div>
          </div>

        </main>

        {/* ПРАВАЯ ПАНЕЛЬ */}
        {isRightPanelOpen && (
          <aside className="w-[450px] bg-white border-l border-gray-200 flex flex-col shrink-0 shadow-[-4px_0_15px_-3px_rgba(0,0,0,0.05)] z-10">

            <div className="h-14 px-4 border-b border-gray-200 flex items-center justify-between bg-white shrink-0">

              <div className="flex items-center gap-2 font-medium text-sm text-gray-800 truncate pr-4">
                <IconFileText className="w-4 h-4 text-[#0b1e4b] shrink-0" />

                <span className="truncate">
                  {activeDocument ? activeDocument.title : 'Документ'}
                </span>
              </div>

              <div className="flex items-center gap-1 text-gray-500">
                <button
                  onClick={() => setIsRightPanelOpen(false)}
                  className="p-1.5 hover:bg-gray-100 rounded"
                >
                  <IconX className="w-4 h-4" />
                </button>
              </div>

            </div>

            <div className="flex border-b border-gray-200 px-4">
              <button
                onClick={() => setActiveTab('excerpt')}
                className={`py-3 px-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'excerpt'
                    ? 'border-[#f95c19] text-[#f95c19]'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Выдержка
              </button>

              <button
                onClick={() => setActiveTab('full')}
                className={`py-3 px-2 text-sm font-medium border-b-2 transition-colors ml-4 ${
                  activeTab === 'full'
                    ? 'border-[#f95c19] text-[#f95c19]'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Весь документ
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 bg-white">

              {isDocumentLoading ? (
                <div className="text-sm text-gray-500 text-center mt-10">
                  Загрузка документа…
                </div>
              ) : documentError ? (
                <div className="text-sm text-red-600 text-center mt-10">
                  {documentError}
                </div>
              ) : activeTab === 'excerpt' ? (
                <div className="text-sm text-gray-800 leading-relaxed font-serif">

                  {activeExcerpt ? (
                    <div className="bg-[#FFD8C2] p-4 rounded-md text-gray-900 border border-[#FBC5A5] whitespace-pre-wrap">
                      {activeExcerpt}
                    </div>
                  ) : (
                    <div className="text-gray-500 text-center mt-10 font-sans">
                      Нажмите на источник под сообщением, чтобы увидеть выдержку.
                    </div>
                  )}

                </div>
              ) : (
                <div className="text-sm text-gray-800 leading-relaxed font-serif whitespace-pre-wrap">
                  {activeDocument
                    ? activeDocument.content
                    : (
                      <div className="text-center text-gray-500 mt-10 font-sans">
                        Нажмите на источник под ответом, чтобы открыть документ.
                      </div>
                    )
                  }
                </div>
              )}

            </div>
          </aside>
        )}

      </div>
    </div>
  );
}
