import sys
from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QComboBox,
    QLabel
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import ollama



impiccated = """
___
|   |
|   O
|  /|\
|  / \
|       
|__
"""



class ChatWorker(QThread):
    """
    Worker thread to handle streaming chat responses from the API.
    """
    update_signal = pyqtSignal(str)  # Signal to update GUI with new text
    finished_signal = pyqtSignal(str)  # Signal when response is complete

    def __init__(self, model, memory):
        super().__init__()
        self.model = model
        self.memory = memory
        self.role = "user"

    def run(self):
        out = ""
        try:
            stream = ollama.chat(
                model=self.model,
                messages=self.memory,
                stream=True,
            )
            for chunk in stream:
                word = chunk['message']['content']
                out += word
                self.update_signal.emit(word)  # Emit each chunk to update GUI
        except Exception as e:
            out = f"Error: {str(e)}"
        self.finished_signal.emit(out)  # Emit the final response


class ChatApp(QWidget):
    def __init__(self,memory=None):
        super().__init__()
        self.setWindowTitle("Ollama Chatbot")
        self.resize(700, 500)
        self.memory = memory if memory else []
        self.worker = None
        self.selected_model = "llama2-uncensored"  # Default model

        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Model selection
        self.model_layout = QHBoxLayout()
        self.model_label = QLabel("Select Model:")
        self.model_label.setStyleSheet("font: 14px;")
        self.model_layout.addWidget(self.model_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.setStyleSheet("font: 14px;")
        self.model_dropdown.addItems(self.get_available_models())  # Populate models
        self.model_dropdown.currentTextChanged.connect(self.change_model)
        self.model_layout.addWidget(self.model_dropdown)

        self.role_dropdown = QComboBox()
        self.role_dropdown.setStyleSheet("font: 14px;")
        self.role_dropdown.addItems(["user","system","assistant"])  # Populate models
        self.role_dropdown.currentTextChanged.connect(self.change_role)
        self.model_layout.addWidget(self.role_dropdown)
        self.role_dropdown.setCurrentIndex(0)


        self.purge_button = QPushButton("Purge Memory")
        self.purge_button.setStyleSheet("font: 14px;")
        self.purge_button.clicked.connect(self.purge_memory)
        self.model_layout.addWidget(self.purge_button)

        self.layout.addLayout(self.model_layout)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font: 14px;")
        self.layout.addWidget(self.chat_display)

        # Input layout
        self.input_layout = QHBoxLayout()

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your message here...")
        self.input_box.setStyleSheet("font: 14px;")
        self.input_box.returnPressed.connect(self.handle_send)  # Send on Enter
        self.input_layout.addWidget(self.input_box)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("font: 14px;")
        self.send_button.clicked.connect(self.handle_send)
        self.input_layout.addWidget(self.send_button)

        self.layout.addLayout(self.input_layout)

        self.input_box.setFocus()  # Set focus back to the input box

    def get_available_models(self):
        """
        Retrieve the list of available models from the Ollama API.
        """
        try:
            models = ollama.list()["models"]
            print([model["name"] for model in models])
            return [model["name"] for model in models]
        except Exception as e:
            print(f"Error retrieving models: {e}")
            return ["llama2.1"]  # Fallback model if there's an error

    def change_model(self, model_name):
        """
        Update the selected model.
        """
        self.selected_model = model_name

    def change_role(self, role):
        """
        Update the selected model.
        """
        self.role = role
        print(f"new role: {self.role}")


    def purge_memory(self):
        """
        Clear the memory list, effectively resetting the conversation history.
        """
        self.memory.clear()
        self.chat_display.clear()
        self.chat_display.append("Memory has been purged.")

    def handle_send(self):
        """
        Handle sending user input and initiate API response streaming.
        """
        user_input = self.input_box.text().strip()

        role = ["user", "system", "assistant"][self.role_dropdown.currentIndex()]
        print(f"{role}: {user_input}")

        if not user_input:
            return

        # Append user message to memory and update display
        self.memory.append({'role': role, 'content': user_input})
        self.chat_display.append(f"{role}: {user_input.strip()}\n")
        self.input_box.clear()

        if role == "user":  # Ensure AI responds only when role is 'user'
            self.chat_display.insertPlainText("\nAssistant: ")

            # Disable input during response generation
            self.input_box.setDisabled(True)
            self.send_button.setDisabled(True)

            # Initialize worker thread for API call
            self.worker = ChatWorker(model=self.selected_model, memory=self.memory)
            self.worker.update_signal.connect(self.update_response_live)
            self.worker.finished_signal.connect(self.finalize_response)
            self.worker.start()
        else:
            # Skip AI response for other roles
            self.chat_display.insertPlainText("")



            
    def update_response_live(self, chunk):
        """
        Update the assistant's response in real-time without creating new lines.
        """
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(chunk)
        self.chat_display.setTextCursor(cursor)  # Ensure cursor stays at the end
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def finalize_response(self, full_response):
        """
        Finalize the assistant's response and update memory.
        """
        self.memory.append({'role': 'assistant', 'content': full_response.strip()})

        # add a blank line after the assistant's text
        self.chat_display.append("")

        # Re-enable input after response generation
        self.input_box.setDisabled(False)
        self.send_button.setDisabled(False)
        self.input_box.setFocus()  # Set focus back to the input box


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp([])
    window.show()
    sys.exit(app.exec_())
