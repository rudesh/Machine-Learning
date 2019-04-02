import tkinter as Tk
from datetime import datetime, timedelta
from typing import Callable, List

from communication.chat import Chat, Message
from util.client import Client
from util.peers import Peers

MAIN_FRAME_COLOR = '#F0F4F8'
DARKER_UI_COLOR = "#D5D9DE"
MESSAGE_RELATED_COLOR = "#F1F1F4"
HIGHLIGHT_COLOR = "#C8EDFC"
WHITE = "white"

LARGE_FONT = "Arial 16"
SMALLER_FONT = "Arial 14"

X, Y = 1000, 500


class VerticalScrolledFrame(Tk.Frame):
    def __init__(self, parent, *args, **kw):
        Tk.Frame.__init__(self, parent, *args, **kw)

        # Create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Tk.Scrollbar(self, orient=Tk.VERTICAL)
        vscrollbar.pack(fill=Tk.Y, side=Tk.RIGHT, expand=Tk.FALSE)
        canvas = Tk.Canvas(self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set)
        canvas.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=Tk.TRUE)
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior, anchor=Tk.NW)

        # Track changes to the canvas and frame width and sync them,
        # Also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


class EntryWithPlaceholder(Tk.Text):
    def __init__(self, master, *args, **kw):
        Tk.Text.__init__(self, master, *args, **kw)

        self.placeholder = 'Type a message to send'
        self.placeholder_color = 'grey'
        self.default_fg_color = 'black'

        self.bind('<FocusIn>', self.foc_in)
        self.bind('<FocusOut>', self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(Tk.END, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('1.0', Tk.END)
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if self.get('1.0', Tk.END) == '\n':
            self.put_placeholder()


class Contact:
    active_chat = None

    def __init__(self,
                 peer: Client,
                 contacts_frame_list: VerticalScrolledFrame,
                 root: Tk.Tk,
                 ensure_chat: Callable[[Client], Chat],
                 unread_count: int):

        self.name = peer.display_str
        if unread_count > 0:
            self.name += f" ({unread_count})"
        self.peer = peer
        self.contactsFrameList = contacts_frame_list
        self.root = root
        self.ensure_chat = ensure_chat
        diff = datetime.now() - peer.last_active
        text_color = 'black' if diff < timedelta(minutes=5) else 'gray'

        self.button = Tk.Label(contacts_frame_list.interior, text=self.name, font=LARGE_FONT, bg=MAIN_FRAME_COLOR,
                               fg=text_color,
                               anchor='w',
                               padx=10, pady=10)
        self.button.bind('<Button-1>', self.chat)
        self.button.pack(fill=Tk.X)

    def update_message_area(self, msg_area: Tk.Text, messages: List[Message]):
        """
        Update conservation field based on the message list.
        """
        msg_area.config(state=Tk.NORMAL)
        msg_area.delete("1.0", Tk.END)
        msg_area.insert(Tk.INSERT, "\n".join([f"{msg.owner.name}: {msg.message_text}" for msg in messages]))
        msg_area.yview_moveto(1.0)
        msg_area.config(state=Tk.DISABLED)

    def chat(self, event):
        if Contact.active_chat:
            Contact.active_chat.update_chat = None

        # Delete previous chat frames
        if len(self.root.winfo_children()) != 1:
            self.root.winfo_children()[1].destroy()

        # Highlight current contact
        for contact in self.contactsFrameList.interior.winfo_children():
            contact.configure(bg=MAIN_FRAME_COLOR)

        self.button.configure(bg=HIGHLIGHT_COLOR)

        chat_frame = Tk.Frame(self.root, bg=WHITE)
        chat_frame.pack(fill=Tk.Y, expand=1)

        Tk.Label(chat_frame, text=self.name, font=LARGE_FONT, width=50, bg=WHITE, anchor='w').pack(pady=10, padx=10)
        Tk.Frame(chat_frame, bg=DARKER_UI_COLOR, height=1).pack(fill=Tk.X)

        general_chat_component = Tk.Frame(chat_frame, bg=MESSAGE_RELATED_COLOR, width=40)

        # Component for holding all the conversations between two parties.
        messages_area = Tk.Text(general_chat_component, height=22, state=Tk.DISABLED)
        messages_area.pack()

        general_chat_component.pack(pady=15, side=Tk.BOTTOM)

        chat = self.ensure_chat(self.peer)
        chat.unread_count = 0
        self.button.configure(text=self.peer.display_str)
        Contact.active_chat = chat
        self.update_message_area(messages_area, chat.msg_list)

        # React to message receive
        chat.update_chat = lambda: self.update_message_area(messages_area, chat.msg_list)

        # Component for message writing
        message_insertion_box = EntryWithPlaceholder(general_chat_component,
                                                     font=SMALLER_FONT,
                                                     bg=MESSAGE_RELATED_COLOR,
                                                     bd=0,
                                                     height=1,
                                                     width=52,
                                                     padx=5,
                                                     pady=5)
        message_insertion_box.pack(side=Tk.LEFT)

        def handle_message_send(event=None):
            """
            Use this function to get message from text box, send it, clear text_box.
            """
            self.root.focus()
            msg = message_insertion_box.get("1.0", Tk.END).strip()
            chat.send_message(msg)
            message_insertion_box.delete("1.0", Tk.END)
            message_insertion_box.focus()

        # Photo taken from here: https://www.brandeps.com/icon/S/Send-03
        self.btn_photo = Tk.PhotoImage(file='Send-03.gif')
        self.btn_photo = self.btn_photo.subsample(4, 4)
        send = Tk.Button(general_chat_component, text='Send', width=50, height=20, command=handle_message_send,
                         image=self.btn_photo)
        self.root.bind('<Return>', handle_message_send)  # binds enter key to handle_message_send
        send.pack(side=Tk.RIGHT, fill=Tk.Y)


class UserInterface:
    ensure_chat: Callable[[Client], Chat]

    def __init__(self, username):
        # Configure
        self.main = Tk.Tk()
        self.main.title('Epyks')
        self.main.configure(background=WHITE)
        self.main.resizable(False, False)
        self.main.geometry(f"{X}x{Y}")

        main_frame = Tk.Frame(self.main, bg=MAIN_FRAME_COLOR)
        main_frame.pack(side=Tk.LEFT, fill=Tk.Y)

        Tk.Frame(main_frame, bg=DARKER_UI_COLOR, width=1).pack(fill=Tk.Y, side=Tk.RIGHT)
        Tk.Label(main_frame, text=username, font=LARGE_FONT, width=25, bg=MAIN_FRAME_COLOR, anchor='w').pack(pady=10,
                                                                                                             padx=10)
        Tk.Frame(main_frame, bg=DARKER_UI_COLOR, height=1).pack(fill=Tk.X)
        Tk.Label(main_frame, text='Contacts', font=LARGE_FONT, bg=MAIN_FRAME_COLOR).pack(pady=10, padx=10)

        self.contacts_frame = VerticalScrolledFrame(main_frame)
        self.contacts_frame.pack(fill=Tk.BOTH, expand=1)

    def run(self):
        self.main.mainloop()

    def update_contacts(self, clients: Peers, chats):
        # Update all the contact list
        for child in self.contacts_frame.interior.winfo_children():
            child.destroy()

        for client in sorted(clients.get_client_list(), key=lambda client: client.last_active, reverse=True):
            unread_count = chats[client.uuid].unread_count if client.uuid in chats else 0
            Contact(client, self.contacts_frame, self.main, self.ensure_chat, unread_count)
