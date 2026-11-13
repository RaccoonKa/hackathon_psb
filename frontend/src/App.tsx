import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  IconPlus, IconSearch, IconBell, IconMessage, IconMore,
  IconPaperclip, IconSend, IconFileText, IconUpload,
  IconDownload, IconMaximize, IconX, IconChevronLeft,
  IconChevronRight, IconCheckCheck, IconMinus, IconBot
} from './Icons';

// Импорт логотипа (Убедись, что файл logo.png лежит в src/assets/)
import PsbLogo from './assets/logo.png';

// --- ИНТЕРФЕЙСЫ БЭКЕНДА ---
interface SourceCitation {
  document_id: string;
  version_id: string | null;
  document_title: string;
  provision_id: string | null;
  structure_path: string[];
  excerpt: string;
  char_start: number | null;
  char_end: number | null;
  score: number | null;
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
  intent: string | null;
  refused: boolean;
  clarification_needed: boolean;
  created_at: string;
}

// --- ЛОКАЛЬНЫЕ ИНТЕРФЕЙСЫ ДЛЯ UI ---
interface Message {
  id: string;
  role: 'user' | 'assistant';
  time: string;
  text: string;
  bullets?: string[];
  footer?: string;
  source?: string;
  sourceId?: string;
  excerpt?: string;
}

export default function App() {
  // Состояния для чатов и сообщений
  const [chats, setChats] = useState<BackendChat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [searchQuery, setSearchQuery] = useState("");

  // Состояния для интерфейса
  const [isRightPanelOpen, setIsRightPanelOpen] = useState(true);
  const [activeChatMenu, setActiveChatMenu] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'excerpt' | 'full'>('excerpt');
  const [inputText, setInputText] = useState("");
  
  // Состояния для загрузки и правой панели
  const [isSending, setIsSending] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [activeDocument, setActiveDocument] = useState<any>(null);
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
    const init = async () => {
      try {
        const res = await axios.get('/api/chats');
        setChats(res.data.items);
        if (res.data.items.length > 0) {
          setActiveChatId(res.data.items[0].id);
        } else {
          handleCreateChat();
        }
      } catch (error) {
        console.error("Ошибка инициализации чатов", error);
      }
    };
    init();
  }, []);

  useEffect(() => {
    if (activeChatId) {
      fetchMessages(activeChatId);
    } else {
      setMessages([]);
    }
  }, [activeChatId]);

  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); };
  useEffect(() => { scrollToBottom(); }, [messages, isSending]);

  // --- API ЗАПРОСЫ ---
  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    try {
      const res = await axios.get('/api/chats', { params: { q: query } });
      setChats(res.data.items);
    } catch (error) {
      console.error("Ошибка поиска", error);
    }
  };

  const handleCreateChat = async () => {
    try {
      const res = await axios.post('/api/chats', {});
      setChats(prev => [res.data, ...prev]);
      setActiveChatId(res.data.id);
      setSearchQuery(""); // Сбрасываем поиск при создании
    } catch (error) {
      console.error("Ошибка создания чата", error);
    }
  };

  const handleDeleteChat = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await axios.delete(`/api/chats/${chatId}`);
      setChats(prev => prev.filter(c => c.id !== chatId));
      if (activeChatId === chatId) {
        setActiveChatId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error("Ошибка удаления", error);
    }
    setActiveChatMenu(null);
  };

  const fetchMessages = async (chatId: string) => {
    try {
      const res = await axios.get(`/api/chats/${chatId}/messages`);
      const adapted = res.data.messages.map((msg: BackendMessage) => ({
        id: msg.id,
        role: msg.role,
        time: formatTime(msg.created_at),
        text: msg.content,
        source: msg.sources?.[0]?.document_title,
        sourceId: msg.sources?.[0]?.document_id,
        excerpt: msg.sources?.[0]?.excerpt
      }));
      setMessages(adapted);
    } catch (error) {
      console.error("Ошибка загрузки сообщений", error);
    }
  };

  // --- ОТПРАВКА СООБЩЕНИЯ ---
  const handleSend = async () => {
    if (!inputText.trim() || !activeChatId || isSending) return;
    
    const currentInput = inputText;
    setInputText("");
    setIsSending(true);
    setLoadingStep(0);

    const stepInterval = setInterval(() => {
      setLoadingStep(prev => (prev < 3 ? prev + 1 : prev));
    }, 1500);

    try {
      const res = await axios.post(`/api/chats/${activeChatId}/messages`, {
        content: currentInput
      });
      
      clearInterval(stepInterval);
      
      const userMsg: BackendMessage = res.data.user_message;
      const aiMsg: BackendMessage = res.data.assistant_message;

      setMessages(prev => [
        ...prev, 
        {
          id: userMsg.id, role: 'user', time: formatTime(userMsg.created_at), text: userMsg.content
        },
        {
          id: aiMsg.id, role: 'assistant', time: formatTime(aiMsg.created_at), text: aiMsg.content,
          source: aiMsg.sources?.[0]?.document_title,
          sourceId: aiMsg.sources?.[0]?.document_id,
          excerpt: aiMsg.sources?.[0]?.excerpt
        }
      ]);
      
      // Обновляем список чатов, чтобы подтянуть новое авто-название
      const chatsRes = await axios.get('/api/chats', { params: { q: searchQuery } });
      setChats(chatsRes.data.items);
      
    } catch (error: any) {
      clearInterval(stepInterval);
      console.error("Ошибка отправки", error);
      let errorMsg = "Ошибка подключения к ИИ.";
      if (error.response?.status === 503) errorMsg = "Сервис ИИ не настроен (проверьте .env на бэкенде).";
      if (error.response?.status === 502) errorMsg = "Ошибка на стороне ИИ (возможно, лимит запросов).";
      
      setMessages(prev => [...prev, {
         id: Date.now().toString(), role: 'assistant', time: formatTime(new Date().toISOString()), text: `⚠ ${errorMsg}`
      }]);
    } finally {
      setIsSending(false);
      setLoadingStep(0);
    }
  };

  // --- ОТКРЫТИЕ ДОКУМЕНТА В ПРАВОЙ ПАНЕЛИ ---
  const openSource = async (docId?: string, excerpt?: string) => {
    setIsRightPanelOpen(true);
    setActiveTab('excerpt');
    setActiveExcerpt(excerpt || null);
    
    if (docId && (!activeDocument || activeDocument.document_id !== docId)) {
      try {
        const res = await axios.get(`/api/documents/${docId}`);
        setActiveDocument(res.data);
      } catch (error) {
        console.error("Не удалось загрузить текст документа", error);
      }
    }
  };

  const formatTime = (isoString: string) => {
    try { return new Date(isoString).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }); } 
    catch { return '14:32'; }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#F8F9FA] font-sans text-gray-800 overflow-hidden">
      
      {/* ШАПКА */}
      <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 shrink-0 z-20">
        <div className="flex items-center gap-4">
          <img src={PsbLogo} alt="ПСБ" className="h-9 object-contain" />
        </div>
        <div className="flex items-center gap-4 text-[#0b1e4b]">
          <button className="p-2 hover:bg-gray-100 rounded-full text-gray-500"><IconBell className="w-5 h-5" /></button>
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
              <IconPlus className="w-4 h-4" /> Новый чат
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
            <p className="px-2 text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">История чатов</p>
            {chats.map((chat) => (
              <div 
                key={chat.id} 
                onClick={() => setActiveChatId(chat.id)}
                className={`group relative flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer text-sm transition-colors ${activeChatId === chat.id ? 'bg-[#152f60] border-l-2 border-[#f95c19]' : 'hover:bg-[#152f60]'}`}
                onMouseLeave={() => setActiveChatMenu(null)}
              >
                <div className="flex items-center gap-2 overflow-hidden whitespace-nowrap">
                  <IconMessage className={`w-4 h-4 shrink-0 ${activeChatId === chat.id ? 'text-[#f95c19]' : 'text-gray-400'}`} />
                  <span className={`truncate ${activeChatId === chat.id ? 'text-white' : 'text-gray-300'}`}>{chat.title}</span>
                </div>
                
                <button 
                  onClick={(e) => { e.stopPropagation(); setActiveChatMenu(activeChatMenu === chat.id ? null : chat.id); }}
                  className={`p-1 rounded hover:bg-[#1e3f7a] text-gray-400 hover:text-white transition-opacity ${activeChatMenu === chat.id || activeChatId === chat.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
                >
                  <IconMore className="w-4 h-4" />
                </button>

                {activeChatMenu === chat.id && (
                  <div className="absolute right-0 top-10 w-40 bg-white rounded-md shadow-lg border border-gray-100 py-1 z-50 text-gray-800">
                    <button className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100">Переименовать</button>
                    <button onClick={(e) => handleDeleteChat(chat.id, e)} className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-red-600">Удалить</button>
                  </div>
                )}
              </div>
            ))}
            {chats.length === 0 && searchQuery && (
              <div className="text-xs text-gray-500 text-center mt-4">Ничего не найдено</div>
            )}
          </div>

          <div className="p-4 mt-auto">
             <div className="border border-dashed border-[#2a4785] rounded-xl p-4 flex flex-col items-center justify-center text-center cursor-pointer hover:bg-[#152f60] transition-colors group">
                <IconUpload className="w-6 h-6 text-gray-400 group-hover:text-[#f95c19] mb-2 transition-colors" />
                <p className="text-xs text-gray-300 font-medium">Загрузить документы</p>
                <p className="text-[10px] text-gray-500 mt-1">PDF, TXT, DOCX</p>
             </div>
          </div>
        </aside>

        {/* ЦЕНТРАЛЬНАЯ ЧАСТЬ */}
        <main className="flex-1 flex flex-col min-w-0 bg-[#F8F9FA]">
          <div className="h-14 px-6 flex items-center border-b border-gray-200 bg-[#F8F9FA] shrink-0">
            <h2 className="font-semibold text-lg text-[#0b1e4b] flex items-center gap-2">
              Юридический помощник 
              <span className="bg-[#E0E7FF] text-[#3730A3] text-xs px-2 py-0.5 rounded-full font-bold">AI</span>
            </h2>
          </div>

          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-4 max-w-[85%] ${msg.role === 'user' ? 'self-end flex-row-reverse' : 'self-start'}`}>
                <div className="shrink-0 mt-1">
                  {msg.role === 'user' ? (
                     <div className="w-8 h-8 rounded-full bg-[#0b1e4b] text-white flex items-center justify-center font-bold text-xs">ВЫ</div>
                  ) : (
                     <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center shadow-sm overflow-hidden p-1">
                       <IconBot className="w-5 h-5 text-[#f95c19]" />
                     </div>
                  )}
                </div>

                <div className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`p-4 rounded-2xl ${msg.role === 'user' ? 'bg-[#D6E4FF] text-[#0b1e4b] rounded-tr-sm' : 'bg-white border border-gray-200 text-gray-800 shadow-sm rounded-tl-sm'}`}>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                    {msg.bullets && (
                      <ul className="list-disc ml-4 mt-2 space-y-1 text-sm">
                        {msg.bullets.map((b, i) => <li key={i}>{b}</li>)}
                      </ul>
                    )}
                    {msg.footer && <p className="text-sm mt-3">{msg.footer}</p>}
                  </div>
                  
                  <div className="flex items-center gap-3 mt-2 text-xs text-gray-500 px-1">
                    {msg.source && (
                      <button onClick={() => openSource(msg.sourceId, msg.excerpt)} className="flex items-center gap-1 text-[#0b1e4b] hover:text-[#f95c19] transition-colors bg-gray-100 hover:bg-orange-50 px-2 py-1 rounded max-w-[300px]">
                        <IconFileText className="w-3 h-3 shrink-0" /> 
                        <span className="truncate">Источник: {msg.source}</span>
                      </button>
                    )}
                    <span className="flex items-center gap-1 ml-auto">
                      {msg.time} 
                      {msg.role === 'user' && <IconCheckCheck className="w-3 h-3 text-blue-500" />}
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
                    <div key={idx} className={`flex items-center gap-2 text-sm transition-opacity duration-500 ${idx > loadingStep ? 'opacity-30' : 'opacity-100'}`}>
                      {idx < loadingStep ? (
                        <span className="text-green-500 text-xs w-3">✓</span>
                      ) : idx === loadingStep ? (
                        <span className="w-3 h-3 border-2 border-[#f95c19] border-t-transparent rounded-full animate-spin shrink-0"></span>
                      ) : (
                        <span className="w-3"></span>
                      )}
                      <span className={idx === loadingStep ? 'text-[#0b1e4b] font-medium' : 'text-gray-500'}>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 bg-[#F8F9FA] border-t border-gray-200">
            <div className="max-w-4xl mx-auto bg-white border border-gray-300 rounded-2xl flex items-center p-2 shadow-sm focus-within:border-[#0b1e4b] focus-within:ring-1 focus-within:ring-[#0b1e4b] transition-all">
              <button className="p-2 text-gray-400 hover:text-[#0b1e4b] rounded-full hover:bg-gray-100 transition-colors">
                <IconPaperclip className="w-5 h-5" />
              </button>
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
                <span className="truncate">{activeDocument ? activeDocument.title : 'ГК РФ. Статья 506–524.pdf'}</span>
              </div>
              <div className="flex items-center gap-1 text-gray-500">
                <button className="p-1.5 hover:bg-gray-100 rounded"><IconDownload className="w-4 h-4" /></button>
                <button className="p-1.5 hover:bg-gray-100 rounded"><IconMaximize className="w-4 h-4" /></button>
                <button onClick={() => setIsRightPanelOpen(false)} className="p-1.5 hover:bg-gray-100 rounded"><IconX className="w-4 h-4" /></button>
              </div>
            </div>

            <div className="flex border-b border-gray-200 px-4">
              <button onClick={() => setActiveTab('excerpt')} className={`py-3 px-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'excerpt' ? 'border-[#f95c19] text-[#f95c19]' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>Выдержка</button>
              <button onClick={() => setActiveTab('full')} className={`py-3 px-2 text-sm font-medium border-b-2 transition-colors ml-4 ${activeTab === 'full' ? 'border-[#f95c19] text-[#f95c19]' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>Весь документ</button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 bg-white">
              {activeTab === 'excerpt' ? (
                 <div className="text-sm text-gray-800 leading-relaxed font-serif">
                    {activeExcerpt ? (
                       <div className="bg-[#FFD8C2] p-4 rounded-md text-gray-900 border border-[#FBC5A5] whitespace-pre-wrap">
                         {activeExcerpt}
                       </div>
                    ) : (
                       <div className="text-gray-500 text-center mt-10 font-sans">Нажмите на источник под сообщением, чтобы увидеть выдержку.</div>
                    )}
                 </div>
              ) : (
                 <div className="text-sm text-gray-800 leading-relaxed font-serif whitespace-pre-wrap">
                    {activeDocument ? activeDocument.content : <div className="text-center text-gray-500 mt-10 font-sans">Документ не загружен</div>}
                 </div>
              )}
            </div>

            <div className="h-16 border-t border-gray-200 px-4 flex items-center justify-between bg-gray-50 shrink-0">
               <div className="flex items-center bg-white border border-gray-200 rounded-lg px-2 py-1 shadow-sm opacity-50 pointer-events-none">
                 <button className="p-1 text-gray-400 hover:text-gray-700"><IconChevronLeft className="w-4 h-4" /></button>
                 <span className="text-xs font-medium px-4">1 из 3</span>
                 <button className="p-1 text-gray-400 hover:text-gray-700"><IconChevronRight className="w-4 h-4" /></button>
               </div>
               <div className="flex items-center bg-white border border-gray-200 rounded-lg px-2 py-1 shadow-sm gap-2 text-gray-600 opacity-50 pointer-events-none">
                 <button className="p-1 hover:text-gray-900"><IconMinus className="w-3 h-3" /></button>
                 <span className="text-xs font-medium w-10 text-center">100%</span>
                 <button className="p-1 hover:text-gray-900"><IconPlus className="w-3 h-3" /></button>
                 <div className="w-px h-4 bg-gray-300 mx-1"></div>
                 <button className="p-1 hover:text-gray-900"><IconMaximize className="w-3 h-3" /></button>
               </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}