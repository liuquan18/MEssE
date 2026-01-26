! Copyright (c) 2024 The YAC Authors
!
! SPDX-License-Identifier: BSD-3-Clause

PROGRAM toy_spmd

  USE mpi
  USE yac
  USE toy_atm, ONLY : main_atm
  USE toy_ocn, ONLY : main_ocn

  IMPLICIT NONE

  INTEGER :: ierror

  ! Initial communicator splitting
  INTEGER :: group_comms(2), yac_comm, toy_comm
  CHARACTER(len=YAC_MAX_CHARLEN) :: group_names(2)
  INTEGER :: comm_rank, comm_size
  INTEGER :: role
  INTEGER, PARAMETER :: IS_ATM = 1
  INTEGER, PARAMETER :: IS_OCN = 2

  ! Initialise MPI
  CALL MPI_Init(ierror)

  group_names(1) = "yac"
  group_names(2) = "spmd"

  ! Generate communicator for YAC and SPMD toy
  ! TODO
  CALL yac_fmpi_handshake(MPI_COMM_WORLD, group_names, group_comms)
  ! END TODO

  yac_comm = group_comms(1)
  toy_comm = group_comms(2)

  ! Initial YAC using YAC communicator
  ! TODO
  CALL yac_finit_comm(yac_comm)
  ! END TODO

  ! Determine role of process
  CALL MPI_Comm_rank(toy_comm, comm_rank, ierror)
  CALL MPI_Comm_size(toy_comm, comm_size, ierror)
  IF (comm_size < 2) THEN
    PRINT *, "too few processes"
    CALL MPI_Abort(toy_comm, -1, ierror)
  END IF
  role = MERGE(IS_ATM, IS_OCN, comm_rank < comm_size / 2)

  ! Run the model in there respective roles
  SELECT CASE(role)
    CASE (IS_ATM)
      CALL main_atm(yac_comm)
    CASE (IS_OCN)
      CALL main_ocn(yac_comm)
    CASE DEFAULT
      PRINT *, "invalid role"
      CALL MPI_Abort(toy_comm, -1, ierror)
  END SELECT

  ! Finalize YAC
  ! TODO
  CALL yac_ffinalize()
  ! END TODO

  ! Finalise MPI
  CALL MPI_Finalize(ierror)

END PROGRAM toy_spmd
