# Copyright (c) 2024 The YAC Authors
#
# SPDX-License-Identifier: BSD-3-Clause

F90 = mpif90
LD = mpif90
FCLAGS = -O0 -g $(shell pkg-config yac-mci --cflags)
LDFLAGS = $(FCLAGS) -Wl,--allow-multiple-definition
LIBS = $(shell pkg-config yac-utils --libs) \
       $(shell pkg-config yac-mci --libs)
MAKEMOD = $(F90) $(FCFLAGS) -fsyntax-only -c

CURL = curl

ICON_GRIDS = \
    grids/icon_grid_0030_R02B03_G.nc \
    grids/icon_grid_0036_R02B04_O.nc \
    grids/icon_grid_0043_R02B04_G.nc
TEST_DATA_URL = http://icon-downloads.mpimet.mpg.de/grids/public/mpim

TOYS = toy_atm.x toy_ocn.x toy_spmd.x

all: $(TOYS) grid_files

grid_files: grids $(ICON_GRIDS)

grids:
	mkdir grids

%.nc:
	@url="$(TEST_DATA_URL)/`echo $(notdir $@) | cut -c 11-14`/$(notdir $@)"; \
	cmd="$(CURL) -sL '$$url' -o $@"; \
	echo "$$cmd"; \
	eval "$$cmd"

toy_atm.mod: toy_common.mod
toy_ocn.mod: toy_common.mod
toy_spmd.mod: toy_atm.mod toy_ocn.mod

%.mod: %.F90
	$(MAKEMOD) $(FCLAGS) $*.F90

toy_spmd.o: toy_atm.mod toy_ocn.mod

%.o: %.mod %.F90
	$(F90) $(FCLAGS) -c $*.F90

toy_atm.x: toy_common.o
toy_ocn.x: toy_common.o
toy_spmd.x: toy_spmd.o toy_atm.o toy_ocn.o toy_common.o

%.x: %.o
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f *.x *.o *.mod metadata.yaml *.nc

neat: clean
	rm -rf grids
