#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
# MODIFIED: Added per-group .so build support for multi-threaded kernel compilation

SHELL := /usr/bin/env bash
MAKEFALGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR := ../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../build)
OBJDIR := $(BUILDDIR)/obj/device
OPSDIR := $(BUILDDIR)/ops

MANIFEST := $(OBJDIR)/manifest
DEVGLUE_OBJ  := $(OBJDIR)/device_glue.o
DEVGLUE_GROUP := $(OBJDIR)/device_glue_group.o  # lightweight glue for per-group .so

INCFLAGS  = -I. -I.. -I$(BUILDDIR)/include -I../include
NVCUFLAGS += $(INCFLAGS) --compiler-options "-fPIC -fvisibility=hidden"
CXXFLAGS  += $(INCFLAGS)

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cu = $(NVCC) $(NVCUFLAGS) -dc $2 -o $1
COMPILE.cc = $(CXX) $(CXXFLAGS) -c $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(suffix $2),$1,$2)
endef

DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -M -dc $1
DEPENDS.cc = $(CXX) $(CXXFLAGS) -M -c $1
define DEPENDS
@$(SAY) "Dependencies" $2;\
 mkdir -p $(dir $1);\
 mk=$$($(call DEPENDS$(suffix $2),$2));\
 [[ $$mk =~ ^[^:]*:(.*)$$ ]];\
 files=$${BASH_REMATCH[1]};\
 files=$$(for x in $$files; do case "$$x" in '\\'|$$'\t') ;; *) echo "$$x"; esac; done);\
 files=$$(for x in $$files; do [[ "$$(realpath "$$x")" == "$$(realpath "$(NCCLDIR)")"* ]] && echo "$$x"; done);\
 echo "$(patsubst %.d,%.o,$1) $1: " $$files > $1
endef

all: $(MANIFEST) ops

ifeq (1,1)
# Case if the <gensrc> directory is generated on-demand:
$(OBJDIR)/gensrc: generate.py
	@mkdir -p $@
	(which python3 >/dev/null || \
	  (bar='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'; \
	   printf "\n$${bar}\nERROR: Building NCCL requires a Python 3 installation invokable as 'python3'.\n$${bar}\n\n" 1>&2; \
	   exit 1)) \
	&& ./generate.py $@ "$(ONLY_FUNCS)"
else
$(OBJDIR)/gensrc:
	@mkdir -p $(OBJDIR); ln -srfn ./gen $@
endif

$(OBJDIR)/gensrc/rules.mk: $(OBJDIR)/gensrc ;
$(OBJDIR)/gensrc/group_build.mk: $(OBJDIR)/gensrc ;
$(OBJDIR)/gensrc/register_rules.mk: $(OBJDIR)/gensrc ;

-include $(OBJDIR)/gensrc/rules.mk
-include $(OBJDIR)/gensrc/group_build.mk
-include $(OBJDIR)/gensrc/register_rules.mk

SRCS = common.cu onerank.cu

LIB_OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS)) $(LIB_OBJS_GEN)

$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/genobj/%.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/%.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/$*)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(OBJDIR)/genobj/%.d: $(OBJDIR)/gensrc/%
	$(call DEPENDS,$@,$<)

# Full device glue (for traditional static linking, still works)
$(DEVGLUE_OBJ): $(LIB_OBJS)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

# Minimal device glue for main libnccl.so (common + onerank only)
$(OBJDIR)/device_glue_min.o: $(OBJDIR)/common.cu.o $(OBJDIR)/onerank.cu.o
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

# Lightweight device glue for per-group .so (only common + onerank)
COMMON_OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS))
$(DEVGLUE_GROUP): $(COMMON_OBJS)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

# Manifest for main libnccl.so: only common + onerank + minimal glue
# (per-op objects go into their own .so, not here)
$(MANIFEST): $(OBJDIR)/common.cu.o $(OBJDIR)/onerank.cu.o $(OBJDIR)/device_glue_min.o
	@echo $^ > $@

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(OBJDIR)/genobj/*.d)

.PHONY: clean ops
clean:
	rm -rf $(OBJDIR) $(OPSDIR)
