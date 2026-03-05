# ==============================================================================
# Compilatore e opzioni di compilazione
# ==============================================================================
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -w -DFF_BOUNDED_BUFFER

# ==============================================================================
# Percorsi di include (-I)
#   - Header di sistema
#   - Dipendenze installate localmente (onnxruntime, ecc.)
#   - WindFlow/FastFlow (directory in HOME)
#   - Directory corrente (header locali del progetto)
# ==============================================================================
INCLUDES := -I /usr/include \
            -I /usr/local/include \
            -I /usr/local/onnxruntime/include \
            -I $(HOME)/WindFlow \
            -I $(HOME)/WindFlow/wf \
            -I $(HOME)/fastflow \
            -I .

# ==============================================================================
# Percorsi delle librerie (-L)
# ==============================================================================
LDFLAGS  := -L /usr/local/onnxruntime/lib \
            -L /usr/local/lib

# ==============================================================================
# Librerie da linkare (-l)
# ==============================================================================
LIBS     := -lonnxruntime -ltensorflow -lrdkafka++ -lrdkafka -lpthread

# ==============================================================================
# Librerie a runtime
# ==============================================================================
RPATH    := -Wl,-rpath,/usr/local/onnxruntime/lib \
            -Wl,-rpath,/usr/local/lib

# ==============================================================================
# Target e sorgenti
# ==============================================================================
TARGET   := windflow-dp
SRCS     := windflow-dp.cpp
# Nota: Essendo gli altri file degli Header (.hpp), non serve compilarli separatamente come .o

# ==============================================================================
# Destinazione per il deploy (copia dell'eseguibile compilato)
# ==============================================================================
DEPLOY_PATH := $(HOME)/crayfish23/experiments-driver/scripts/

.PHONY: all clean deploy

# Target di default
all: $(TARGET)

# Compilazione dell'eseguibile
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LDFLAGS) $(LIBS) $(RPATH)

# Pulizia dei file generati
clean:
	rm -f $(TARGET)

# Build + copia automatica nella cartella di deploy
deploy: $(TARGET)
	cp $(TARGET) $(DEPLOY_PATH)
	@echo "[SUCCESS] $(TARGET) compilato e copiato in $(DEPLOY_PATH)"
