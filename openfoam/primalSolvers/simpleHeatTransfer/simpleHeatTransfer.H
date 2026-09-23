/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2007-2023 PCOpt/NTUA
    Copyright (C) 2013-2023 FOSS GP
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::simpleHeatTransfer

Description
    Incompressible steady-state (SIMPLE) primal solver for the adjoint
    optimisation framework, extended with a scalar energy (temperature)
    equation so that temperature-dependent viscosity models
    (e.g. powerLawArrhenius) can be used.

    This is a copy of Foam::simple with an added temperature field T solved by
    a convection-diffusion equation with a viscous-dissipation source:

        div(phi, T) - laplacian(DT, T) = (1/cp) (tau : grad(U))

    with tau = nu (grad(U) + grad(U)^T). DT (thermal diffusivity) and cp
    (specific heat) are read from constant/transportProperties.

    T is constructed and registered *before* the incompressibleVars (and hence
    the laminar transport / viscosity model) so that the viscosity model can
    look it up during its own construction. The energy equation is solved each
    SIMPLE iteration right before laminarTransport().correct(), which then
    recomputes nu from the updated T and strain rate (full primal coupling).

SourceFiles
    simpleHeatTransfer.C

\*---------------------------------------------------------------------------*/

#ifndef simpleHeatTransfer_H
#define simpleHeatTransfer_H

#include "incompressiblePrimalSolver.H"
#include "SIMPLEControl.H"
#include "IOMRFZoneList.H"
#include "objective.H"
#include "IOdictionary.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class simpleHeatTransfer Declaration
\*---------------------------------------------------------------------------*/

class simpleHeatTransfer
:
    public incompressiblePrimalSolver
{
private:

    // Private Member Functions

        //- No copy construct
        simpleHeatTransfer(const simpleHeatTransfer&) = delete;

        //- No copy assignment
        void operator=(const simpleHeatTransfer&) = delete;


protected:

    // Protected data

        //- Solver control
        autoPtr<SIMPLEControl> solverControl_;

        //- Transport properties (read locally to grab DT and cp). Not
        //- registered to avoid clashing with the copy created by
        //- incompressibleVars / singlePhaseTransportModel.
        IOdictionary transportProperties_;

        //- Temperature field.
        //  Declared (and therefore constructed/registered) BEFORE incoVars_
        //  so the viscosity model can look it up at construction time.
        volScalarField T_;

        //- Thermal diffusivity [m2/s]
        dimensionedScalar DT_;

        //- Specific heat (kinematic) [m2/s2/K]
        dimensionedScalar cp_;

        //- Reference to incompressibleVars
        //  Used for convenience and to avoid repetitive dynamic_casts
        //  Same as getIncoVars()
        incompressibleVars& incoVars_;

        //- MRF zones
        IOMRFZoneList MRF_;

        //- Cumulative continuity error
        scalar cumulativeContErr_;

        //- List of objectives related to this primal solver
        UPtrList<objective> objectives_;

        //- Allow the execution of function objects in optimisation mode.
        bool allowFunctionObjects_;


    //- Protected Member Functions

        //- Allocate incompressibleVars and return reference to be used for
        //- convenience in the rest of the class.
        incompressibleVars& allocateVars();

        //- In case variable names are different than the base ones,
        //- add extra schemes and relaxation factors to the appropriate dicts
        void addExtraSchemes();

        //- Compute continuity errors
        void continuityErrors();

        //- Solve the temperature (energy) equation
        void solveTEqn();


public:

    // Static Data Members

        //- Run-time type information
        TypeName("simpleHeatTransfer");


    // Constructors

        //- Construct from mesh and dictionary
        simpleHeatTransfer
        (
            fvMesh& mesh,
            const word& managerType,
            const dictionary& dict,
            const word& solverName
        );


    //- Destructor
    virtual ~simpleHeatTransfer() = default;


    // Member Functions

        virtual bool readDict(const dictionary& dict);

        // Evolution

            //- Execute one iteration of the solution algorithm
            virtual void solveIter();

            //- Steps to be executed before each main SIMPLE iteration
            virtual void preIter();

            //- The main SIMPLE iter
            virtual void mainIter();

            //- Steps to be executed before each main SIMPLE iteration
            virtual void postIter();

            //- Main control loop
            virtual void solve();

            //- Looper (advances iters, time step)
            virtual bool loop();

            //- Restore initial field values if necessary
            virtual void restoreInitValues();

            //- Functions to be called before loop
            virtual void preLoop();

            //- Functions to be called after loop
            virtual void postLoop();

            //- Write average iteration
            virtual bool writeData(Ostream& os) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
