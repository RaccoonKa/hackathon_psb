import { useState, useRef, useEffect } from 'react';
import {
  IconPlus, IconSearch, IconBell, IconMessage, IconMore,
  IconPaperclip, IconSend, IconFileText, IconUpload,
  IconDownload, IconMaximize, IconX, IconChevronLeft,
  IconChevronRight, IconCheckCheck, IconMinus, IconBot
} from './Icons';

// Импорт логотипа (Убедись, что файл logo.png лежит в src/assets/)
import PsbLogo from './assets/logo.png';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  time: string;
  text: string;
  bullets?: string[];
  footer?: string;
  source?: string;
  sourceId?: string;
}

const MOCK_CHATS = [
  { id: '1', title: 'Условия договора поставки', active: true },
  { id: '2', title: 'Налоговые льготы', active: false },
  { id: '3', title: 'Проверка контрагента', active: false },
  { id: '4', title: 'Изменения в ГК РФ', active: false },
  { id: '5', title: 'Судебная практика по 223-ФЗ', active: false },
];

const MOCK_MESSAGES: Message[] = [
  { id: 'm1', role: 'user', time: '14:32', text: 'Какие основные требования к заключению договора поставки?' },
  {
    id: 'm2', role: 'assistant', time: '14:32',
    text: 'Основные требования к заключению договора поставки регулируются Гражданским кодексом РФ, в частности, статьями 506–524. Ключевые моменты:',
    bullets: ['Четкое определение предмета договора (товары, ассортимент, количество);', 'Согласование цены и порядка расчетов;', 'Условия о сроках и порядке поставки.'],
    footer: 'Также рекомендуется предусмотреть ответственность сторон за нарушение условий договора.',
    source: 'ГК РФ, ст. 506–524', sourceId: 'doc1'
  },
  { id: 'm3', role: 'user', time: '14:33', text: 'А каковы последствия нарушения сроков поставки?' },
  {
    id: 'm4', role: 'assistant', time: '14:33',
    text: 'При нарушении сроков поставки возможны следующие последствия:',
    bullets: ['Взыскание неустойки (штрафа или пени);', 'Возмещение убытков;', 'Расторжение договора в судебном порядке.'],
    source: 'ГК РФ, ст. 523, 524', sourceId: 'doc2'
  }
];

export default function App() {
  const [isRightPanelOpen, setIsRightPanelOpen] = useState(true);
  const [activeChatMenu, setActiveChatMenu] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'excerpt' | 'full'>('excerpt');
  const [messages, setMessages] = useState<Message[]>(MOCK_MESSAGES);
  const [inputText, setInputText] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); };
  useEffect(() => { scrollToBottom(); }, [messages]);

  const handleSend = () => {
    if (!inputText.trim()) return;
    const newMsg: Message = { id: Date.now().toString(), role: 'user', time: '14:35', text: inputText };
    setMessages([...messages, newMsg]);
    setInputText("");
    
    setTimeout(() => {
        setMessages(prev => [...prev, {
            id: Date.now().toString(), role: 'assistant', time: '14:35',
            text: 'Это заглушка ответа. Бэкенд еще не подключен.',
            source: 'Тестовый документ', sourceId: 'doc1'
        }]);
    }, 1000);
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
            <button className="w-full bg-[#f95c19] hover:bg-[#e04a10] text-white py-2.5 px-4 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors shadow-sm">
              <IconPlus className="w-4 h-4" /> Новый чат
            </button>
          </div>
          
          <div className="px-4 pb-2">
             <div className="relative">
                <IconSearch className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
                <input type="text" placeholder="Поиск по чатам..." className="w-full bg-[#152f60] text-sm text-white placeholder-gray-400 rounded-md py-2 pl-9 pr-3 outline-none focus:ring-1 focus:ring-[#f95c19]" />
             </div>
          </div>

          <div className="flex-1 overflow-y-auto px-2 mt-2 space-y-1">
            <p className="px-2 text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">История чатов</p>
            {MOCK_CHATS.map((chat) => (
              <div 
                key={chat.id} 
                className={`group relative flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer text-sm transition-colors ${chat.active ? 'bg-[#152f60] border-l-2 border-[#f95c19]' : 'hover:bg-[#152f60]'}`}
                onMouseLeave={() => setActiveChatMenu(null)}
              >
                <div className="flex items-center gap-2 overflow-hidden whitespace-nowrap">
                  <IconMessage className={`w-4 h-4 ${chat.active ? 'text-[#f95c19]' : 'text-gray-400'}`} />
                  <span className={chat.active ? 'text-white' : 'text-gray-300'}>{chat.title}</span>
                </div>
                
                <button 
                  onClick={(e) => { e.stopPropagation(); setActiveChatMenu(activeChatMenu === chat.id ? null : chat.id); }}
                  className={`p-1 rounded hover:bg-[#1e3f7a] text-gray-400 hover:text-white transition-opacity ${activeChatMenu === chat.id || chat.active ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
                >
                  <IconMore className="w-4 h-4" />
                </button>

                {activeChatMenu === chat.id && (
                  <div className="absolute right-0 top-10 w-40 bg-white rounded-md shadow-lg border border-gray-100 py-1 z-50 text-gray-800">
                    <button className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100">Закрепить</button>
                    <button className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100">Переименовать</button>
                    <button className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-red-600">Удалить</button>
                  </div>
                )}
              </div>
            ))}
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
                    <p className="text-sm leading-relaxed">{msg.text}</p>
                    {msg.bullets && (
                      <ul className="list-disc ml-4 mt-2 space-y-1 text-sm">
                        {msg.bullets.map((b, i) => <li key={i}>{b}</li>)}
                      </ul>
                    )}
                    {msg.footer && <p className="text-sm mt-3">{msg.footer}</p>}
                  </div>
                  
                  <div className="flex items-center gap-3 mt-2 text-xs text-gray-500 px-1">
                    {msg.source && (
                      <button onClick={() => setIsRightPanelOpen(true)} className="flex items-center gap-1 text-[#0b1e4b] hover:text-[#f95c19] transition-colors bg-gray-100 hover:bg-orange-50 px-2 py-1 rounded">
                        <IconFileText className="w-3 h-3" /> Источник: {msg.source}
                      </button>
                    )}
                    <span className="flex items-center gap-1">
                      {msg.time} 
                      {msg.role === 'user' && <IconCheckCheck className="w-3 h-3 text-blue-500" />}
                    </span>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 bg-[#F8F9FA] border-t border-gray-200">
            <div className="max-w-4xl mx-auto bg-white border border-gray-300 rounded-2xl flex items-center p-2 shadow-sm focus-within:border-[#0b1e4b] focus-within:ring-1 focus-within:ring-[#0b1e4b] transition-all">
              <button className="p-2 text-gray-400 hover:text-[#0b1e4b] rounded-full hover:bg-gray-100 transition-colors">
                <IconPaperclip className="w-5 h-5" />
              </button>
              <input 
                type="text" value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Задайте свой вопрос..." 
                className="flex-1 bg-transparent outline-none px-2 text-sm text-gray-700 placeholder-gray-400"
              />
              <button onClick={handleSend} className="bg-[#f95c19] hover:bg-[#e04a10] text-white p-2 rounded-xl transition-colors ml-2 flex items-center justify-center">
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
                <span className="truncate">ГК РФ. Статья 506–524.pdf</span>
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
                    <h3 className="text-xl font-bold mb-4 font-sans">Статья 506. Договор поставки</h3>
                    <p className="mb-4">1. По договору поставки поставщик обязуется передать в обусловленный срок или сроки производимые или закупаемые им товары покупателю для использования в предпринимательской деятельности или в иных целях, не связанных с личным, семейным, домашним и иным подобным использованием.</p>
                    <div className="bg-[#FFD8C2] p-3 -mx-3 rounded-md mb-4 text-gray-900 border border-[#FBC5A5]">
                      2. К договору поставки применяются общие положения о купле-продаже, если иное не установлено правилами настоящей главы.
                    </div>
                    <p>3. Договор поставки может быть заключен как на поставку определенных товаров, так и на поставку товаров, указанных родовыми признаками.</p>
                 </div>
              ) : (
                 <div className="text-sm text-gray-500 text-center mt-10">Здесь будет полный текст документа...</div>
              )}
            </div>

            <div className="h-16 border-t border-gray-200 px-4 flex items-center justify-between bg-gray-50 shrink-0">
               <div className="flex items-center bg-white border border-gray-200 rounded-lg px-2 py-1 shadow-sm">
                 <button className="p-1 text-gray-400 hover:text-gray-700"><IconChevronLeft className="w-4 h-4" /></button>
                 <span className="text-xs font-medium px-4">1 из 3</span>
                 <button className="p-1 text-gray-400 hover:text-gray-700"><IconChevronRight className="w-4 h-4" /></button>
               </div>
               <div className="flex items-center bg-white border border-gray-200 rounded-lg px-2 py-1 shadow-sm gap-2 text-gray-600">
                 <button className="p-1 hover:text-gray-900"><IconMinus className="w-3 h-3" /></button>
                 <span className="text-xs font-medium w-10 text-center">100%</span>
                 {/* Здесь теперь обычный плюс без квадрата */}
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