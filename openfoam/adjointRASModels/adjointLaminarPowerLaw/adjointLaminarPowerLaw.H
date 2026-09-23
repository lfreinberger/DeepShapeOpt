/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2007-2023 PCOpt/NTUA
    Copyright (C) 2013-2023 FOSS GP
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
    Foam::incompressibleAdjoint::adjointRASModels::adjointLaminarPowerLaw

Description
    Laminar adjoint "turbulence" model that ADDS the shear-rate viscosity
    linearization for a powerLaw (shear-thinning) primal viscosity to the
    otherwise frozen-viscosity adjoint.

    Temperature is held frozen in the adjoint (no adjoint energy equation,
    no dnu/dT term); only the shear-rate dependence is linearized.

    Derivation (continuous adjoint, kinematic form). The primal viscous term
    is div(2 nu(gammaDot) S) with S = symm(grad U) and
    gammaDot = sqrt(2 S:S). Linearising nu w.r.t. the velocity adds, to the
    adjoint momentum equation, the term

        - div( beta (S:grad(Ua)) S ),   beta = (4/gammaDot) dnu/dgammaDot

    For the power law nu = k gammaDot^(n-1) exp(E/(R T)) (unclamped branch),
    dnu/dgammaDot = (n-1) nu / gammaDot, hence

        beta = 4 (n-1) nu / gammaDot^2.

    beta is masked to zero wherever nu is on a clamp (nu == nuMin or nuMax),
    where dnu/dgammaDot = 0. The term is returned (explicitly, using the
    current Ua) through adjointMeanFlowSource(), which adjointSimple adds to
    the adjoint momentum equation.

    Dictionary entries (adjointRASProperties), optional sub-dict
    adjointLaminarPowerLawCoeffs:
        linearizeViscosity     true;  // toggle both terms (default true)
        linearizationSign      1;     // +1 / -1; +1 verified, see below
        includeFISensitivity   true;  // toggle the shape-gradient term
        FISensitivityTranspose false; // transpose convention of that term
    The power-law n and the clamps nuMin/nuMax are read from
    constant/transportProperties -> powerLawArrheniusCoeffs.

    The adjoint FIELD term above has been verified by a duality (dot-product)
    test: with a volumetric momentum source added to the primal only, the
    exact adjoint satisfies dJ/deps = -int Ua.f dV. On a tapered-channel case
    with a velocity-uniformity objective and n = 0.5 the frozen-viscosity
    adjoint recovers only 0.7% of the true gradient, this model with
    linearizationSign 1 recovers it to within the discretisation error, which
    halves under mesh refinement (1.391 / 1.166 / 1.083 on 60x30 / 120x60 /
    240x120). linearizationSign -1 gives ~0 and is decisively wrong.

    In addition to the adjoint field, a shear-rate-dependent viscosity also
    depends on the shape directly through the grid metrics; that contribution
    enters the shape gradient (not the adjoint equations) and is supplied by
    FISensitivityTerm() below. It is NOT a small correction: on the wall-bump
    verification case (openfoam/verification/) at n = 0.5 the shape gradient
    goes from 0.6883 of the true value with the frozen adjoint, to 0.8294 with
    the adjoint-field linearisation alone, to 1.0022 with this term added --
    against a Newtonian harness control of 0.9997. Both terms are therefore
    verified. The frozen temperature costs a further 0.45%.

    FISensitivityTranspose is NOT resolved by that case: gradU & S is nearly
    symmetric in a 2-D shear-dominated flow (the two conventions differ by
    4e-6). Re-check it if this model is ever used on a strongly 3-D flow.

SourceFiles
    adjointLaminarPowerLaw.C

\*---------------------------------------------------------------------------*/

#ifndef adjointLaminarPowerLaw_H
#define adjointLaminarPowerLaw_H

#include "adjointLaminar.H"
#include "Switch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressibleAdjoint
{
namespace adjointRASModels
{

/*---------------------------------------------------------------------------*\
                    Class adjointLaminarPowerLaw Declaration
\*---------------------------------------------------------------------------*/

class adjointLaminarPowerLaw
:
    public adjointLaminar
{
private:

    // Private data

        //- Power-law index (from transportProperties)
        scalar n_;

        //- Viscosity clamps (from transportProperties), used to mask the
        //  linearization where the primal viscosity is saturated
        scalar nuMin_;
        scalar nuMax_;

        //- Toggle the shear-rate viscosity linearization
        Switch linearizeViscosity_;

        //- Sign multiplier to match the solver's adjoint sign convention
        scalar linearizationSign_;

        //- Include the direct shape-sensitivity contribution of dnu/dgammaDot
        //  (FISensitivityTerm). Set false to reproduce the pre-2026-07
        //  behaviour, where the term was missing from the shape gradient.
        Switch includeFISensitivity_;

        //- Return the transpose of FISensitivityTerm(). The default (false)
        //  is verified only up to the near-symmetry of gradU & S in 2-D shear
        //  flow; see the class documentation.
        Switch FISensitivityTranspose_;


    // Private Member Functions

        //- No copy construct
        adjointLaminarPowerLaw(const adjointLaminarPowerLaw&) = delete;

        //- No copy assignment
        void operator=(const adjointLaminarPowerLaw&) = delete;

        //- Read the power-law coefficients and the linearization controls
        void readCoeffs();

        //- Shared building blocks of both linearization terms:
        //  S = symm(grad U), beta = 4(n-1)nu/gammaDot^2 (clamp-masked) and
        //  c = S : grad(Ua)
        void linearizationFields
        (
            tmp<volSymmTensorField>& tS,
            tmp<volScalarField>& tbeta,
            tmp<volScalarField>& tc
        ) const;


public:

    //- Runtime type information
    TypeName("adjointLaminarPowerLaw");


    // Constructors

        //- Construct from components
        adjointLaminarPowerLaw
        (
            incompressibleVars& primalVars,
            incompressibleAdjointMeanFlowVars& adjointVars,
            objectiveManager& objManager,
            const word& adjointTurbulenceModelName
                = adjointTurbulenceModel::typeName,
            const word& modelName = typeName
        );


    //- Destructor
    virtual ~adjointLaminarPowerLaw() = default;


    // Member Functions

        //- Source terms to the adjoint momentum equation due to the
        //- differentiation of the shear-rate-dependent viscosity
        virtual tmp<volVectorField> adjointMeanFlowSource();

        //- Direct (grid-metric) shape dependence of the shear-rate-dependent
        //- viscosity; "Term 6" of computeGradDxDbMultiplier
        virtual tmp<volTensorField> FISensitivityTerm();

        //- Read adjointRASProperties dictionary
        virtual bool read();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace adjointRASModels
} // End namespace incompressibleAdjoint
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
