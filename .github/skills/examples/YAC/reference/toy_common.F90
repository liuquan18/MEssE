! Copyright (c) 2024 The YAC Authors
!
! SPDX-License-Identifier: BSD-3-Clause

MODULE toy_common

  USE, INTRINSIC :: iso_c_binding
  USE mpi
  USE yac
  USE yac_utils, ONLY : &
    yac_read_icon_grid_information_parallel_c, yac_free_c

  IMPLICIT NONE

  PUBLIC :: read_icon_grid
  PUBLIC :: define_fields
  PUBLIC :: send_field
  PUBLIC :: receive_field
  PUBLIC :: max_char_length
  PUBLIC :: nsteps

  INTEGER, PARAMETER :: max_char_length = 132
  INTEGER, PARAMETER :: nsteps = 5

  CONTAINS

  SUBROUTINE read_icon_grid( &
    grid_filename, comm, cell_to_vertex, x_vertices, y_vertices, &
    x_cells, y_cells, cell_mask, global_cell_id)

    CHARACTER(LEN=max_char_length)             :: grid_filename
    INTEGER                                    :: comm
    INTEGER, ALLOCATABLE, INTENT(OUT)          :: cell_to_vertex(:,:)
    DOUBLE PRECISION, ALLOCATABLE, INTENT(OUT) :: x_vertices(:)
    DOUBLE PRECISION, ALLOCATABLE, INTENT(OUT) :: y_vertices(:)
    DOUBLE PRECISION, ALLOCATABLE, INTENT(OUT) :: x_cells(:)
    DOUBLE PRECISION, ALLOCATABLE, INTENT(OUT) :: y_cells(:)
    INTEGER, ALLOCATABLE, INTENT(OUT)          :: cell_mask(:)
    INTEGER, ALLOCATABLE, INTENT(OUT)          :: global_cell_id(:)

    INTEGER(KIND=C_INT) :: num_vertices_c
    INTEGER(KIND=C_INT) :: num_cells_c
    TYPE(C_PTR)         :: num_vertices_per_cell_c
    TYPE(C_PTR)         :: cell_to_vertex_c
    TYPE(C_PTR)         :: global_cell_id_c
    TYPE(C_PTR)         :: cell_owner_c
    TYPE(C_PTR)         :: global_vertex_ids_c
    TYPE(C_PTR)         :: vertex_owner_c
    TYPE(C_PTR)         :: x_vertices_c
    TYPE(C_PTR)         :: y_vertices_c
    TYPE(C_PTR)         :: x_cells_c
    TYPE(C_PTR)         :: y_cells_c
    TYPE(C_PTR)         :: cell_mask_c

    INTEGER(KIND=c_int), POINTER :: cell_to_vertex_ptr(:,:)
    REAL(KIND=c_double), POINTER :: x_vertices_ptr(:)
    REAL(KIND=c_double), POINTER :: y_vertices_ptr(:)
    REAL(KIND=c_double), POINTER :: x_cells_ptr(:)
    REAL(KIND=c_double), POINTER :: y_cells_ptr(:)
    INTEGER(KIND=c_int), POINTER :: cell_mask_ptr(:)
    INTEGER(KIND=c_int), POINTER :: global_cell_id_ptr(:)

    INTEGER                        :: ierror
    CHARACTER(LEN=max_char_length) :: root_grid_filename

    ! Checks for a consistent grid filename across all processes in comm
    root_grid_filename = grid_filename
    CALL MPI_Bcast( &
      root_grid_filename, max_char_length, MPI_CHARACTER, 0, comm, ierror)
    IF (root_grid_filename /= grid_filename) THEN
      PRINT *, "ERROR(read_icon_grid): inconsistent grid filename ('", &
               TRIM(root_grid_filename), "' /= '", TRIM(grid_filename), "')"
      CALL MPI_Abort(comm, -1, ierror)
    END IF
    CALL MPI_Barrier(comm, ierror)

    ! use YAC utility routine to read grid data in parallel and distribute it
    ! among all processes
    ! (The YAC utility library only contains a iso-c binding interface
    !  for this respective interface and Fortran implementation. This has
    !  to be taken into account when passing arguments to this interface.)
    CALL yac_read_icon_grid_information_parallel_c( &
      TRIM(grid_filename) // c_null_char, INT(comm, YAC_MPI_FINT_KIND), &
      num_vertices_c, num_cells_c, num_vertices_per_cell_c, cell_to_vertex_c, &
      global_cell_id_c, cell_owner_c, global_vertex_ids_c, &
      vertex_owner_c, x_vertices_c, y_vertices_c, x_cells_c, y_cells_c, &
      cell_mask_c)

    ALLOCATE( &
      cell_to_vertex(3, num_cells_c), &
      x_vertices(num_vertices_c), y_vertices(num_vertices_c), &
      x_cells(num_cells_c), y_cells(num_cells_c), cell_mask(num_cells_c), &
      global_cell_id(num_cells_c))

    ! convert c_ptr's to Fortran pointers
    CALL C_F_POINTER(cell_to_vertex_c, cell_to_vertex_ptr, (/3, num_cells_c/))
    CALL C_F_POINTER(x_vertices_c, x_vertices_ptr, (/num_vertices_c/))
    CALL C_F_POINTER(y_vertices_c, y_vertices_ptr, (/num_vertices_c/))
    CALL C_F_POINTER(x_cells_c, x_cells_ptr, (/num_cells_c/))
    CALL C_F_POINTER(y_cells_c, y_cells_ptr, (/num_cells_c/))
    CALL C_F_POINTER(cell_mask_c, cell_mask_ptr, (/num_cells_c/))
    CALL C_F_POINTER(global_cell_id_c, global_cell_id_ptr, (/num_cells_c/))

    ! convert from C to Fortran datatypes and indexing
    cell_to_vertex = INT(cell_to_vertex_ptr) + 1
    x_vertices = DBLE(x_vertices_ptr)
    y_vertices = DBLE(y_vertices_ptr)
    x_cells = DBLE(x_cells_ptr)
    y_cells = DBLE(y_cells_ptr)
    cell_mask = INT(cell_mask_ptr)
    global_cell_id = INT(global_cell_id_ptr) + 1

    ! free memory allocated by the call to
    ! yac_read_icon_grid_information_parallel_c
    ! (These arrays were allocated in C, hence they have to be freed using the
    !  C free routine. yac_free_c is a Fortran interface provided by the YAC
    !  utility library to this routine.)
    CALL yac_free_c(num_vertices_per_cell_c)
    CALL yac_free_c(cell_to_vertex_c)
    CALL yac_free_c(global_cell_id_c)
    CALL yac_free_c(cell_owner_c)
    CALL yac_free_c(global_vertex_ids_c)
    CALL yac_free_c(vertex_owner_c)
    CALL yac_free_c(x_vertices_c)
    CALL yac_free_c(y_vertices_c)
    CALL yac_free_c(x_cells_c)
    CALL yac_free_c(y_cells_c)
    CALL yac_free_c(cell_mask_c)

  END SUBROUTINE read_icon_grid

  SUBROUTINE define_fields( &
    comp_id, point_id, field_taux_id, field_tauy_id, field_sfwflx_id, &
    field_sftemp_id, field_thflx_id, field_iceatm_id, field_sst_id, &
    field_oceanu_id, field_oceanv_id, field_iceoce_id)

    INTEGER, INTENT(IN) :: comp_id
    INTEGER, INTENT(IN) :: point_id

    INTEGER, INTENT(OUT) :: field_taux_id
    INTEGER, INTENT(OUT) :: field_tauy_id
    INTEGER, INTENT(OUT) :: field_sfwflx_id
    INTEGER, INTENT(OUT) :: field_sftemp_id
    INTEGER, INTENT(OUT) :: field_thflx_id
    INTEGER, INTENT(OUT) :: field_iceatm_id
    INTEGER, INTENT(OUT) :: field_sst_id
    INTEGER, INTENT(OUT) :: field_oceanu_id
    INTEGER, INTENT(OUT) :: field_oceanv_id
    INTEGER, INTENT(OUT) :: field_iceoce_id


    ! Define fields
    field_taux_id  =   def_field("surface_downward_eastward_stress", 2, comp_id, point_id)
    field_tauy_id  =   def_field("surface_downward_northward_stress", 2, comp_id, point_id)
    field_sfwflx_id  = def_field("surface_fresh_water_flux", 3, comp_id, point_id)
    field_sftemp_id  = def_field("surface_temperature", 1, comp_id, point_id)
    field_thflx_id  =  def_field("total_heat_flux", 4, comp_id, point_id)
    field_iceatm_id  = def_field("atmosphere_sea_ice_bundle", 4, comp_id, point_id)
    field_sst_id  =    def_field("sea_surface_temperature", 1, comp_id, point_id)
    field_oceanu_id  = def_field("eastward_sea_water_velocity", 1, comp_id, point_id)
    field_oceanv_id  = def_field("northward_sea_water_velocity", 1, comp_id, point_id)
    field_iceoce_id =  def_field("ocean_sea_ice_bundle", 5, comp_id, point_id)

  END SUBROUTINE define_fields

  FUNCTION def_field(name, collection_size, comp_id, point_id)

    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN)          :: collection_size
    INTEGER, INTENT(IN)          :: comp_id
    INTEGER, INTENT(IN)          :: point_id
    INTEGER                      :: def_field

    INTEGER, PARAMETER :: num_point_ids = 1
    INTEGER :: point_ids(num_point_ids)
    CHARACTER(LEN=max_char_length), PARAMETER :: timestep = "1"
    INTEGER, PARAMETER :: timestep_unit = YAC_TIME_UNIT_SECOND

    def_field = -1
    point_ids(1) = point_id

    ! Define field
    ! TODO
    CALL yac_fdef_field ( &
      name, comp_id, point_ids, num_point_ids, collection_size, &
      timestep, timestep_unit, def_field)
    ! END TODO

  END FUNCTION def_field

  SUBROUTINE send_field(field_id, field)

    INTEGER, INTENT(IN)          :: field_id
    DOUBLE PRECISION, INTENT(IN) :: field(:,:)

    INTEGER :: info, ierror, num_points, collection_size

    ! Get field dimensions
    num_points = SIZE(field, 1)
    collection_size = SIZE(field, 2)

    ! Put field
    ! TODO
    CALL yac_fput(field_id, num_points, collection_size, field, info, ierror)
    ! END TODO

  END SUBROUTINE send_field

  SUBROUTINE receive_field(comp_name, field_id, field)

    CHARACTER(LEN=max_char_length) :: comp_name
    INTEGER, INTENT(IN)            :: field_id
    DOUBLE PRECISION, INTENT(OUT)  :: field(:,:)

    INTEGER :: i, info, ierror, num_points, collection_size
    CHARACTER(LEN=max_char_length), PARAMETER :: debug_format = &
      "(A3,' ',A34,I2,' min: ',F8.2,' max: ',F8.2)"
    CHARACTER(LEN=max_char_length) :: field_name

    ! Get field dimensions
    num_points = SIZE(field, 1)
    collection_size = SIZE(field, 2)

    ! Initialise field to zero
    field(:,:) = 0.0d0
    info = YAC_ACTION_NONE

    ! Get field
    ! TODO
    CALL yac_fget(field_id, num_points, collection_size, field, info, ierror)
    ! END TODO

    ! Get field name from field id
    ! TODO
    field_name = yac_fget_field_name(field_id)
    ! END TODO

    IF (info /= YAC_ACTION_NONE) THEN
      ! Print some debugging output
      DO i = 1, collection_size
        WRITE(*, debug_format) &
          comp_name, field_name, i, MINVAL(field(:,i)), MAXVAL(field(:,i))
      END DO
    END IF

  END SUBROUTINE receive_field

END MODULE toy_common
